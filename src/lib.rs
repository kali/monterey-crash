use std::mem;
use std::ptr::NonNull;
pub struct OwnedRepr<A> {
    pub it: Vec<A>,
}
impl<A> OwnedRepr<A> {
    pub(crate) fn as_slice(&self) -> &[A] {
        &self.it
    }
    pub(crate) fn as_ptr(&self) -> *const A {
        self.it.as_ptr()
    }
}
impl<A> Clone for OwnedRepr<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        Self {
            it: self.as_slice().to_owned(),
        }
    }
}
use std::mem::size_of;
use std::mem::MaybeUninit;
pub trait RawData: Sized {
    type Elem;
}
pub unsafe trait RawDataClone: RawData {
    unsafe fn clone_with_ptr(&self, ptr: *const Self::Elem) -> (Self, *const Self::Elem);
}
pub unsafe trait Data: RawData {}
impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
}
unsafe impl<A> Data for OwnedRepr<A> {}
unsafe impl<A> RawDataClone for OwnedRepr<A>
where
    A: Clone,
{
    unsafe fn clone_with_ptr(&self, ptr: *const Self::Elem) -> (Self, *const Self::Elem) {
        let mut u = self.clone();
        let mut new_ptr = u.as_ptr();
        if size_of::<A>() != 0 {
            let our_off = (ptr as isize - self.as_ptr() as isize) / mem::size_of::<A>() as isize;
            new_ptr = new_ptr.offset(our_off);
        }
        (u, new_ptr)
    }
}
pub unsafe trait DataOwned: Data {
    type MaybeUninit: DataOwned<Elem = MaybeUninit<Self::Elem>>;
    fn new(elements: Vec<Self::Elem>) -> Self;
}
unsafe impl<A> DataOwned for OwnedRepr<A> {
    type MaybeUninit = OwnedRepr<MaybeUninit<A>>;
    fn new(elements: Vec<A>) -> Self {
        OwnedRepr { it: elements }
    }
}
use std::ptr;
pub unsafe trait TrustedIterator {}
unsafe impl TrustedIterator for std::ops::Range<usize> {}
pub fn to_vec_mapped<I, F, B>(iter: I, mut f: F) -> Vec<B>
where
    I: TrustedIterator + ExactSizeIterator,
    F: FnMut(I::Item) -> B,
{
    let (size, _) = iter.size_hint();
    let mut result = Vec::with_capacity(size);
    let mut out_ptr = result.as_mut_ptr();
    let mut len = 0;
    iter.fold((), |(), elt| unsafe {
        ptr::write(out_ptr, f(elt));
        len += 1;
        result.set_len(len);
        out_ptr = out_ptr.offset(1);
    });
    result
}
pub struct ArrayBase<A> {
    data: OwnedRepr<A>,
    ptr: *const A,
}
pub type Array<A> = ArrayBase<A>;
impl<A> Clone for ArrayBase<A>
where
    OwnedRepr<A>: RawDataClone<Elem = A>,
{
    fn clone(&self) -> ArrayBase<A> {
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.ptr);
            ArrayBase { data, ptr }
        }
    }
}
impl<A> ArrayBase<A> {
    pub(crate) unsafe fn from_data_ptr(data: OwnedRepr<A>, ptr: *const A) -> Self {
        let array = ArrayBase { data, ptr };
        array
    }
}
impl<A> ArrayBase<A> {
    pub(crate) unsafe fn with_strides_dim(self) -> ArrayBase<A> {
        ArrayBase {
            data: self.data,
            ptr: self.ptr,
        }
    }
}
impl<A> ArrayBase<A> {
    pub fn from_shape_fn<F>(shape: usize, f: F) -> Self
    where
        F: FnMut(usize) -> A,
    {
        let v = to_vec_mapped((0..shape).into_iter(), f);
        unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }
    pub unsafe fn from_shape_vec_unchecked(shape: usize, v: Vec<A>) -> Self {
        Self::from_vec_dim_stride_unchecked(shape, v)
    }
    unsafe fn from_vec_dim_stride_unchecked(shape: usize, mut v: Vec<A>) -> Self {
        let ptr = v.as_mut_ptr();
        ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim()
    }
    fn first(&self) -> &A {
        unsafe { &*(self.ptr as *const A) }
    }
    fn first_mut(&mut self) -> &mut A {
        unsafe { &mut *(self.ptr as *mut A) }
    }
}
type TractResult<T> = Result<T, ()>;
use std::sync::Arc;
#[derive(Copy, Clone)]
enum BinOp {
    Min,
    Max,
}
#[derive(Clone, Copy)]
enum OutputStoreSpec {
    View {
        m_axis: usize,
    },
    Strides {
        col_byte_stride: isize,
        mr: usize,
        nr: usize,
        m: usize,
        n: usize,
    },
}
#[derive(Clone)]
enum ProtoFusedSpec {
    BinScalar(AttrOrInput, BinOp),
    BinPerRow(AttrOrInput, BinOp),
    BinPerCol(AttrOrInput, BinOp),
    AddRowColProducts(AttrOrInput, AttrOrInput),
    AddUnicast(OutputStoreSpec, AttrOrInput),
    Store,
}
#[derive(Clone)]
struct MatMulUnary {}
impl TypedOp for MatMulUnary {}
#[derive(Clone)]
struct TypedSource {}
impl TypedOp for TypedSource {}
trait TypedOp: Send + Sync + 'static {}
#[derive(Clone)]
enum AttrOrInput {
    Attr(Arc<()>),
    Input(usize),
}
trait SpecialOps<O> {
    fn create_source(&self) -> O;
    fn wire_node(&mut self, op: O, inputs: &[OutletId]) -> TractResult<Vec<OutletId>>;
}
struct Graph<O> {
    pub nodes: Vec<Node<O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
}
impl<O> Default for Graph<O> {
    fn default() -> Graph<O> {
        Graph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }
}
impl<O> Graph<O>
where
    Graph<O>: SpecialOps<O>,
{
    pub fn add_source(&mut self) -> TractResult<OutletId> {
        let source = self.create_source();
        let id = self.add_node(source)?;
        let id = OutletId::new(id, 0);
        self.inputs.push(id);
        Ok(id)
    }
}
impl<O> Graph<O> {
    pub fn add_node(&mut self, op: O) -> TractResult<usize> {
        let id = self.nodes.len();
        let node = Node {
            id,
            op,
            inputs: vec![],
        };
        self.nodes.push(node);
        Ok(id)
    }
    pub fn add_edge(&mut self, outlet: OutletId, inlet: InletId) -> TractResult<()> {
        let succ = &mut self.nodes[inlet.node];
        if inlet.slot == succ.inputs.len() {
            succ.inputs.push(outlet);
        }
        Ok(())
    }
}
struct Node<O> {
    pub id: usize,
    pub inputs: Vec<OutletId>,
    #[cfg_attr(feature = "serialize", serde(skip))]
    pub op: O,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct OutletId {
    pub node: usize,
    pub slot: usize,
}
impl OutletId {
    pub fn new(node: usize, slot: usize) -> OutletId {
        OutletId { node, slot }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
struct InletId {
    pub node: usize,
    pub slot: usize,
}
struct ModelPatch<O> {
    pub model: Graph<O>,
    pub inputs: HashMap<usize, usize>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
}
impl<O> Default for ModelPatch<O> {
    fn default() -> ModelPatch<O> {
        ModelPatch {
            model: Graph::default(),
            inputs: HashMap::default(),
            incoming: HashMap::new(),
            shunt_outlet_by: HashMap::new(),
        }
    }
}
impl<O> ModelPatch<O>
where
    Graph<O>: SpecialOps<O>,
{
    pub fn tap_model(&mut self, model: &Graph<O>, outlet: OutletId) -> TractResult<OutletId> {
        let id = self.model.add_source()?;
        self.incoming.insert(id, outlet);
        Ok(id)
    }
    pub fn shunt_outside(&mut self, outlet: OutletId, by: OutletId) -> TractResult<()> {
        self.shunt_outlet_by.insert(outlet, by);
        Ok(())
    }
    pub fn replace_single_op(
        patched_model: &Graph<O>,
        node: &Node<O>,
        inputs: &[OutletId],
        new_op: O,
    ) -> TractResult<ModelPatch<O>> {
        let mut patch = ModelPatch::default();
        let new_op = new_op.into();
        let inputs = inputs
            .iter()
            .map(|i| patch.tap_model(patched_model, *i))
            .collect::<TractResult<Vec<_>>>()?;
        let wires = patch.model.wire_node(new_op, &inputs)?;
        for (ix, o) in wires.iter().enumerate() {
            patch.shunt_outside(OutletId::new(node.id, ix), *o)?;
        }
        Ok(patch)
    }
    pub fn apply(self, target: &mut Graph<O>) -> TractResult<()> {
        let ModelPatch {
            model: patch,
            incoming: mut mapping,
            shunt_outlet_by,
            ..
        } = self;
        let mut all_inputs = HashMap::new();
        let model_input_outlets = target.inputs.clone();
        for node in patch.nodes {
            if target.inputs.contains(&OutletId::new(node.id, 0))
                && mapping.contains_key(&OutletId::new(node.id, 0))
            {
                continue;
            }
            let Node {
                id: patch_node_id,
                inputs,
                op,
            } = node;
            let added_node_id = target.add_node(op)?;
            mapping.insert(
                OutletId::new(patch_node_id, 0),
                OutletId::new(added_node_id, 0),
            );
            all_inputs.insert(added_node_id, inputs);
        }
        for (outlet, by) in shunt_outlet_by {
            let replace_by = mapping[&by];
            for o in target.outputs.iter_mut() {
                if *o == outlet {
                    *o = replace_by;
                }
            }
        }
        for (node, inputs) in all_inputs {
            for (ix, input) in inputs.into_iter().enumerate() {
                target.add_edge(mapping[&input], InletId { node, slot: ix })?;
            }
        }
        target.inputs = model_input_outlets;
        Ok(())
    }
}
type TypedModel = Graph<Box<dyn TypedOp>>;
type TypedModelPatch = ModelPatch<Box<dyn TypedOp>>;
impl SpecialOps<Box<dyn TypedOp>> for TypedModel {
    fn create_source(&self) -> Box<dyn TypedOp> {
        Box::new(TypedSource {})
    }
    fn wire_node(
        &mut self,
        op: Box<dyn TypedOp>,
        inputs: &[OutletId],
    ) -> TractResult<Vec<OutletId>> {
        {
            let id = self.add_node(op)?;
            inputs
                .iter()
                .enumerate()
                .try_for_each(|(ix, i)| self.add_edge(*i, InletId { node: id, slot: ix }))?;
            TractResult::Ok(vec![OutletId::new(id, 0)])
        }
    }
}
use std::collections::HashMap;
fn dump_pfs(pfs: &ProtoFusedSpec) {
    let ptr = pfs as *const ProtoFusedSpec as *const u8;
    for i in 0..std::mem::size_of::<ProtoFusedSpec>() {
        let v = unsafe { *ptr.add(i) };
        if v == 0 {
            print!("__ ");
        } else {
            print!("{:02x} ", v);
        }
        if i % 8 == 7 {
            print!("| ");
        }
    }
    println!("");
}
fn crasher_monterey() {
    let mut model = TypedModel::default();
    let source = model.add_source().unwrap();
    let mm = model
        .wire_node(Box::new(MatMulUnary {}), &[source])
        .unwrap()[0];
    model.outputs = vec![mm];
    let patch = TypedModelPatch::replace_single_op(
        &model,
        &model.nodes[mm.node],
        &[source],
        Box::new(MatMulUnary {}),
    )
    .unwrap();
    patch.apply(&mut model).unwrap();
    let packed_as = Array::from_shape_fn(1, |_| (Box::new(()), vec![ProtoFusedSpec::Store]));
    eprintln!("in ndarray:");
    dump_pfs(&packed_as.first().1[0]);
    let mut cloned = packed_as.clone();
    std::mem::drop(packed_as);
    eprintln!("cloned:");
    dump_pfs(&cloned.first().1[0]);
    unsafe {
        std::ptr::drop_in_place(&mut cloned.first_mut().1[0]);
    }
    eprintln!("Dropped in place");
    std::mem::drop(cloned);
    eprintln!("Clone dropped");
}
#[test]
fn t1() {
    crasher_monterey()
}
