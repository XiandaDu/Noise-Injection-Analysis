`timescale 1ps / 1ps

module layer_norm
#(
    parameter integer       D_W      = 8,                // output data width (int8)
    parameter integer       D_W_ACC  = 32,               // input and coefficient width (int32)
    parameter integer       N        = 768,              // vector length
    parameter signed [D_W_ACC-1:0] N_INV = 1398101,     // (1/N)<<FP_BITS constant
    parameter integer       FP_BITS  = 30,               // fixed-point bits for multiplication
    parameter integer       MAX_BITS = 31                // used for constant dividend
)
(
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       enable,
    input  logic                       in_valid,
    input  logic signed  [D_W_ACC-1:0] qin,    // input vector element
    input  logic signed  [D_W_ACC-1:0] bias,   // bias to be added at the end
    input  logic [$clog2(D_W_ACC)-1:0] shift,  // shift amount (for scaling)
    output logic                       out_valid,
    output logic signed  [D_W_ACC-1:0] qout    // normalized output element
);

  // Constant dividend for division: 1 << MAX_BITS
  localparam integer DIVIDENT = (1 << MAX_BITS);
  // Total pipeline delay needed (accumulation + sqrt and division latency)
  localparam integer PIPE_DELAY = N + 50;

  //--------------------------------------------------------------------------
  // Vector Counter & Initialize signals for accumulators
  //--------------------------------------------------------------------------

  logic [$clog2(N):0] cnt;
  always_ff @(posedge clk) begin
    if (rst) begin
      cnt <= 0;
    end else if (enable && in_valid) begin
      cnt <= (cnt == N-1) ? 0 : cnt + 1;
    end
  end

  // Signal to initialize accumulators at the start of a vector
  logic initialize_acc;
  assign initialize_acc = (cnt == 0) && in_valid;

  //--------------------------------------------------------------------------
  // Accumulate qsum = sum(qin)
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] qsum;
  acc #(
    .D_W(D_W_ACC),
    .D_W_ACC(D_W_ACC)
  ) acc_qsum (
    .clk       (clk),
    .rst       (rst),
    .enable    (enable && in_valid),
    .initialize(initialize_acc),
    .in_data   (qin),
    .result    (qsum)
  );

  //--------------------------------------------------------------------------
  // Compute q_shift = qin >> shift and then qsum_sq = sum(q_shift*q_shift)
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] q_shift;
  assign q_shift = qin >>> shift;  // arithmetic right shift

  logic signed [D_W_ACC-1:0] qsum_sq;
  mac #(
    .D_W(D_W_ACC),
    .D_W_ACC(D_W_ACC)
  ) mac_qsum_sq (
    .clk        (clk),
    .rst        (rst),
    .enable     (enable && in_valid),
    .initialize (initialize_acc),
    .a          (q_shift),
    .b          (q_shift),
    .result     (qsum_sq)
  );

  //--------------------------------------------------------------------------
  // Delay original qin and bias for final per-element computation.
  // The delay must equal the latency from when the vector is input to when
  // the scalar factor is computed.
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] qin_delayed;
  sreg #(
    .D_W   (D_W_ACC),
    .DEPTH (PIPE_DELAY)
  ) sreg_qin (
    .clk      (clk),
    .rst      (rst),
    .shift_en (enable),
    .data_in  (qin),
    .data_out (qin_delayed)
  );

  logic signed [D_W_ACC-1:0] bias_delayed;
  sreg #(
    .D_W   (D_W_ACC),
    .DEPTH (PIPE_DELAY)
  ) sreg_bias (
    .clk      (clk),
    .rst      (rst),
    .shift_en (enable),
    .data_in  (bias),
    .data_out (bias_delayed)
  );

  logic valid_delayed;
  sreg #(
    .D_W   (1),
    .DEPTH (PIPE_DELAY)
  ) sreg_valid (
    .clk      (clk),
    .rst      (rst),
    .shift_en (enable),
    .data_in  (in_valid),
    .data_out (valid_delayed)
  );

  //--------------------------------------------------------------------------
  // Latch the accumulator outputs at the end of each vector
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] qsum_reg, qsum_sq_reg;
  always_ff @(posedge clk) begin
    if (rst) begin
      qsum_reg    <= 0;
      qsum_sq_reg <= 0;
    end else if (enable && (in_valid || valid_delayed) && (cnt == 0)) begin
      qsum_reg    <= qsum;
      qsum_sq_reg <= qsum_sq;
    end
  end

  //--------------------------------------------------------------------------
  // Compute Mean: qmean = (qsum * N_INV) >> FP_BITS
  //--------------------------------------------------------------------------

  logic signed [2*D_W_ACC-1:0] qmul;
  assign qmul = qsum_reg * N_INV;  // 64-bit multiplication
  logic signed [D_W_ACC-1:0] qmean;
  assign qmean = qmul >>> FP_BITS;

  //--------------------------------------------------------------------------
  // Compute qmean_sq = (qmean * qsum_reg) >> (2*shift)
  // and variance = qsum_sq_reg - qmean_sq
  //--------------------------------------------------------------------------

  logic signed [2*D_W_ACC-1:0] qmean_mul;
  assign qmean_mul = qmean * qsum_reg;
  logic signed [D_W_ACC-1:0] qmean_sq;
  assign qmean_sq = qmean_mul >>> (2 * shift);

  logic signed [D_W_ACC-1:0] variance;
  assign variance = qsum_sq_reg - qmean_sq;

  //--------------------------------------------------------------------------
  // Instantiate sqrt module to compute sqrt_out = floor(sqrt(variance))
  // (Assumed latency: 16 cycles)
  //--------------------------------------------------------------------------

  // Generate a one-cycle pulse to indicate variance is ready.
  logic variance_valid;
  logic sqrt_valid;
  logic [15:0] sqrt_out;
  logic sqrt_valid_reg;
  always_ff @(posedge clk) begin
    if (rst) begin
      variance_valid <= 0;
    end else if (enable && in_valid && (cnt == N-1))
      variance_valid <= 1;
    else
      variance_valid <= 0;
    sqrt_valid_reg <= sqrt_valid;
  end


  // Instantiate the sqrt module using its defined port names.
  sqrt #(
    .D_W(D_W_ACC)  // Set D_W to 32 bits.
  ) sqrt_inst (
    .clk      (clk),
    .rst      (rst),
    .enable   (enable),
    .in_valid (variance_valid),
    .qin      (variance),
    .out_valid(sqrt_valid),
    .qout     (sqrt_out)
  );

  //--------------------------------------------------------------------------
  // Compute std = sqrt_out << shift
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] std;
  assign std = $signed({{(D_W_ACC-16){1'b0}}, sqrt_out}) <<< shift;

  //--------------------------------------------------------------------------
  // Instantiate division module to compute factor = floor(DIVIDENT / std)
  // (Division latency: up to 32 cycles)
  //--------------------------------------------------------------------------

  logic div_valid;
  logic signed [D_W_ACC-1:0] factor;
  div #(
    .D_W(D_W_ACC)
  ) div_inst (
    .clk      (clk),
    .rst      (rst),
    .in_valid (sqrt_valid_reg),
    .enable   (enable),
    .divisor  (std),
    .dividend (DIVIDENT),
    .quotient (factor),
    .out_valid(div_valid)
  );

  logic signed [D_W_ACC-1:0] div_count;
  logic signed [D_W_ACC-1:0] factor_delay_depth;
  always_ff @(posedge clk) begin
    if (rst) begin
      div_count <= 0;
    end else if (sqrt_valid_reg) begin
      div_count <= 0;
    end else if (div_valid) begin
      factor_delay_depth <= 29-div_count;
    end else begin
      div_count <= div_count + 1;
    end
    if (factor_delay_depth != 0) begin
      factor_delay_depth <= factor_delay_depth - 1;
    end
  end

  // Latch factor for use in every element of the current vector
  logic signed [D_W_ACC-1:0] factor_hold;
  logic signed [D_W_ACC-1:0] qmean_delay;
  always_ff @(posedge clk) begin
    if (rst) begin
      factor_hold <= 0;
    end else if (div_valid) begin
      factor_hold <= factor;
    end
    if (factor_delay_depth == 0 && div_count <100) begin
      qmean_delay <= qmean;
    end
  end

  //--------------------------------------------------------------------------
  // Compute final per-element normalized result:
  //   r = delayed_qin - qmean;
  //   qout_mul = r * factor_hold;
  //   qout = (qout_mul >> 1) + delayed_bias;
  //--------------------------------------------------------------------------

  logic signed [D_W_ACC-1:0] r_val;
  assign r_val = qin_delayed - qmean_delay;  // subtract scalar mean from each element

  logic signed [2*D_W_ACC-1:0] qout_mul;
  assign qout_mul = r_val * factor_hold;

  logic signed [D_W_ACC-1:0] qout_reg;
  assign qout_reg = (qout_mul >>> 1) + bias_delayed;

  assign qout = qout_reg;

  //--------------------------------------------------------------------------
  // Generate output valid signal (aligned with our delayed pipeline)
  //--------------------------------------------------------------------------

  assign out_valid = valid_delayed;

endmodule