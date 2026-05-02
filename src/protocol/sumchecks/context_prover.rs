use rokoko::{
    common::{config::NOF_BATCHES, ring_arithmetic::RingElement},
    protocol::sumcheck_utils::{
        combiner::Combiner, common::SumcheckBaseData, diff::DiffSumcheck,
        elephant_cell::ElephantCell, linear::LinearSumcheck, product::ProductSumcheck,
        ring_to_field_combiner::RingToFieldCombiner, selector_eq::SelectorEq,
    },
};

pub struct ProverSumcheckContext {
    pub witness_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub witness_conjugated_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // for verifying norms. Should be optional?
    pub main_witness_selector_sumcheck: ElephantCell<SelectorEq<RingElement>>,
    pub projection_selector_sumcheck: Option<ElephantCell<SelectorEq<RingElement>>>,
    pub type1sumcheck: Vec<Type1ProverSumcheckContext>, // for verifying inner evaluation points
    pub type3sumcheck: Option<Type3ProverSumcheckContext>, // for verifying the projection
    pub type31sumchecks: Option<[Type31ProverSumcheckContext; NOF_BATCHES]>, // for verifying the projection
    pub l2sumcheck: Option<L2ProverSumcheckContext>,
    pub linfsumcheck: Option<LinfSumcheckContext>,
    pub vdfsumcheck: Option<VDFProverSumcheckContext>, // for verifying the VDF, only used in the first round
    pub combiner: ElephantCell<Combiner<RingElement>>,
    pub field_combiner: ElephantCell<RingToFieldCombiner>,
    pub next: Option<Box<ProverSumcheckContext>>,
}

// VDF sumcheck: we prove that M · w = b where M is the VDF matrix and b = (-y_0, 0, ..., 0, y_t).
//
// A is a DF_MATRIX_HEIGHT × DF_MATRIX_WIDTH CRS matrix.
// G = I_{DF_MATRIX_HEIGHT} ⊗ g^T is the gadget matrix (block-diagonal binary decomposition).
//   G is DF_MATRIX_HEIGHT × DF_MATRIX_WIDTH, where DF_MATRIX_WIDTH = DF_BITS * DF_MATRIX_HEIGHT.
//   Each block row r of G has g^T = (1, 2, 4, ..., 2^{DF_BITS-1}) in columns [r*DF_BITS..(r+1)*DF_BITS) and zeros elsewhere.
//
// Per DF step, the matrix block involves G and A, each with DF_MATRIX_HEIGHT rows.
// The step stride DF_STRIDE = DF_MATRIX_HEIGHT ensures that A-powers for step i
// overlap with G-powers for step i+1, yielding a telescoping claim.
//
// The full DF matrix has the structure:
//   |------------------|      |-------- |
//   | G                |      | -y_0    |   <- G · w_0 = -y_0  (HEIGHT rows)
//   | A  G             |      |  0      |   <- A · w_0 + G · w_1 = 0  (HEIGHT rows each)
//   |    A  G          |      |  0      |
//   |       A  G       |  w = |  0      |
//   |          ...     |      |  ...    |
//   |           A  G   |      |  0      |
//   |              A   |      |  y_t    |   <- A · w_{last} = y_t  (HEIGHT rows)
//   |------------------|      |-------- |
//
// where y_0, y_t ∈ R^{DF_MATRIX_HEIGHT} and w = (w_0 // w_1) (columns stacked vertically).
//
// We batch all rows with consecutive powers of challenge c.
// DF_STRIDE = HEIGHT, so step i uses G-powers c^{HEIGHT*i}..c^{HEIGHT*i + HEIGHT-1}
// and A-powers c^{HEIGHT*(i+1)}..c^{HEIGHT*(i+1) + HEIGHT-1} (which overlap with G of step i+1).
// This overlap makes intermediate y values telescope in the batched claim.
//
// We factor this into:
//   vdf_batched_row[j] = sum_{r=0}^{HEIGHT-1} c^r · G[r,j] + sum_{r=0}^{HEIGHT-1} c^{HEIGHT+r} · A[r,j]
//     For j in [r*DF_BITS..(r+1)*DF_BITS): G[r,j] = 2^{j - r*DF_BITS}, other G rows are 0.
//     So: vdf_batched_row[j] = c^{j/DF_BITS} · 2^{j%DF_BITS} + sum_{r=0}^{HEIGHT-1} c^{HEIGHT+r} · A[r,j]
//   vdf_step_powers[i] = c^{DF_STRIDE * i}  for i = 0..2K-1
//
// So the batched relation becomes:
//   (vdf_step_powers ⊗ vdf_batched_row) · w = sum_{r=0}^{HEIGHT-1} c^r · (-y_0[r]) + c^{DF_STRIDE*2K+r} · y_t[r]
//
// For the verifier:
//   - MLE[vdf_batched_row] evaluation is a small sumcheck (DF_MATRIX_WIDTH elements)
//   - MLE[vdf_step_powers] evaluation is efficient via the tensor structure:
//     vdf_step_powers[i] = c^{DF_STRIDE*i}, so MLE = prod_k ((1-x_k) + x_k · c^{DF_STRIDE * 2^{n-1-k}})
//     with iterative squaring: c_power starts at c^{DF_STRIDE} and squares each step.
pub struct VDFProverSumcheckContext {
    pub vdf_step_powers_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub vdf_batched_row_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct L2ProverSumcheckContext {
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct LinfSumcheckContext {
    pub all_one_constant_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

pub struct Type1ProverSumcheckContext {
    pub inner_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub outer_evaluation_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

// we want to show
// <c2 \otimes c0 \otimes j_batched, witness> = batched_projection
pub struct Type31ProverSumcheckContext {
    pub c_2_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // across columns
    pub c_0_sumcheck: ElephantCell<LinearSumcheck<RingElement>>, // across blocks
    pub j_batched_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub output: ElephantCell<ProductSumcheck<RingElement>>,
}

// we want to check that
// (I \otimes J) · witness = projected_witness
// post batching, this can be written as
// c^T (I \otimes J) · witness c_2 = c^T projected_witness c_2
// To be more precise, the projected witness is vectorised (by stacking columns)
// witness itself is vertically aligned so it can be viewed as a single column so we write:
// (c_2 \otimes c)^T (I \otimes J) · witness = (c_2 \otimes c)^T projected_witness
// we keep c and c_2 separated as c_2 will be needed as ``outer evaluation point'' since the  prover will open to
// c^T (I \otimes J) · witness and (c_2 \otimes c)^T · projected_witness and verify consistency between the two using the outer evaluation point c_2.
// c = (c_0, c_1) so that c^T (I \otimes J) = c_0 \otimes c_1^T J
// c_1^T J is denoted as flattened_projection_matrix
// to sum up, the relation we prove via sumcheck
// (c_2 \otimes c_0 \otimes c_1^T J) · witness = (c_2 \otimes c_0 \otimes c_1)^T projected_witness
pub struct Type3ProverSumcheckContext {
    pub c2l_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c0l_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub flattened_projection_matrix_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c2r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c0r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub c1r_sumcheck: ElephantCell<LinearSumcheck<RingElement>>,
    pub lhs: ElephantCell<ProductSumcheck<RingElement>>,
    pub rhs: ElephantCell<ProductSumcheck<RingElement>>,
    pub output: ElephantCell<DiffSumcheck<RingElement>>,
}

impl ProverSumcheckContext {
    pub fn partial_evaluate_all(&mut self, r: &RingElement) {
        self.witness_sumcheck.borrow_mut().partial_evaluate(r);
        self.witness_conjugated_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        self.main_witness_selector_sumcheck
            .borrow_mut()
            .partial_evaluate(r);
        if let Some(sumcheck) = self.projection_selector_sumcheck.as_ref() {
            sumcheck.borrow_mut().partial_evaluate(r)
        }
        for type1 in &mut self.type1sumcheck {
            type1
                .inner_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type1
                .outer_evaluation_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
        if let Some(type3) = &mut self.type3sumcheck {
            type3
                .flattened_projection_matrix_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            type3.c0r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c1r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c2r_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c0l_sumcheck.borrow_mut().partial_evaluate(r);
            type3.c2l_sumcheck.borrow_mut().partial_evaluate(r);
        }

        if let Some(type31sumchecks) = &mut self.type31sumchecks {
            for type31 in type31sumchecks {
                type31.c_2_sumcheck.borrow_mut().partial_evaluate(r);
                type31.c_0_sumcheck.borrow_mut().partial_evaluate(r);
                type31.j_batched_sumcheck.borrow_mut().partial_evaluate(r);
            }
        }

        // it's dumb, but it doesn't do anything except reducing the degree
        if let Some(linf) = &mut self.linfsumcheck {
            linf.all_one_constant_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }

        if let Some(vdf) = &mut self.vdfsumcheck {
            vdf.vdf_step_powers_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
            vdf.vdf_batched_row_sumcheck
                .borrow_mut()
                .partial_evaluate(r);
        }
    }
}
