use std::mem;
use std::ptr::NonNull;
pub struct OwnedRepr<A> {
    it: Vec<A>,
}
impl<A> OwnedRepr<A> {
    pub(crate) fn from(v: Vec<A>) -> Self {
        Self { it: v }
    }
    pub(crate) fn as_slice(&self) -> &[A] {
        &self.it
    }
    pub(crate) fn as_ptr(&self) -> *const A {
        self.it.as_ptr()
    }
    pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
        NonNull::new(self.it.as_mut_ptr()).unwrap()
    }
}
impl<A> Clone for OwnedRepr<A>
where
    A: Clone,
{
    fn clone(&self) -> Self {
        Self::from(self.as_slice().to_owned())
    }
}
use std::mem::size_of;
use std::mem::MaybeUninit;
pub unsafe trait RawData: Sized {
    type Elem;
}
pub unsafe trait RawDataClone: RawData {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>);
}
pub unsafe trait Data: RawData {
    fn into_owned(self_: ArrayBase<Self>) -> Array<Self::Elem>
    where
        Self::Elem: Clone;
    fn try_into_owned_nocopy(self_: ArrayBase<Self>) -> Result<Array<Self::Elem>, ArrayBase<Self>>;
}
unsafe impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
}
unsafe impl<A> Data for OwnedRepr<A> {
    fn into_owned(self_: ArrayBase<Self>) -> Array<Self::Elem>
    where
        A: Clone,
    {
        unimplemented!()
    }
    fn try_into_owned_nocopy(self_: ArrayBase<Self>) -> Result<Array<Self::Elem>, ArrayBase<Self>> {
        unimplemented!()
    }
}
unsafe impl<A> RawDataClone for OwnedRepr<A>
where
    A: Clone,
{
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>) {
        let mut u = self.clone();
        let mut new_ptr = u.as_nonnull_mut();
        if size_of::<A>() != 0 {
            let our_off =
                (ptr.as_ptr() as isize - self.as_ptr() as isize) / mem::size_of::<A>() as isize;
            new_ptr = NonNull::new(new_ptr.as_ptr().offset(our_off)).unwrap();
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
        OwnedRepr::from(elements)
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

pub struct ArrayBase<S>
where
    S: RawData,
{
    data: S,
    ptr: std::ptr::NonNull<S::Elem>,
}
pub type Array<A> = ArrayBase<OwnedRepr<A>>;
impl<S: RawDataClone> Clone for ArrayBase<S> {
    fn clone(&self) -> ArrayBase<S> {
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.ptr);
            ArrayBase { data, ptr }
        }
    }
}
impl<A, S> ArrayBase<S>
where
    S: RawData<Elem = A>,
{
    pub(crate) unsafe fn from_data_ptr(data: S, ptr: NonNull<A>) -> Self {
        let array = ArrayBase { data, ptr };
        array
    }

}
impl<A, S> ArrayBase<S>
where
    S: RawData<Elem = A>,
{
    pub(crate) unsafe fn with_strides_dim(self) -> ArrayBase<S> {
        ArrayBase {
            data: self.data,
            ptr: self.ptr,
        }
    }
}
impl<S, A> ArrayBase<S>
where
    S: DataOwned<Elem = A>,
{
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
        let ptr = std::ptr::NonNull::new(v.as_mut_ptr()).unwrap();
        ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim()
    }
    fn first(&self) -> &A {
unsafe {		&*(self.ptr.as_ptr() as *const A) }
	}
    fn first_mut(&mut self) -> &mut A {
unsafe {		&mut*(self.ptr.as_mut() as *mut A) }
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
trait TypedOp: Send + Sync + 'static {
}
#[derive(Clone)]
enum AttrOrInput {
    Attr(Arc<()>),
    Input(usize),
}
trait SpecialOps<O> {
    fn create_source(&self) -> O;
    fn wire_node(&mut self, op: O, inputs: &[OutletId]) -> TractResult<Vec<OutletId>>;
}
struct Graph<O>
{
    pub nodes: Vec<Node<O>>,
    pub inputs: Vec<OutletId>,
    pub outputs: Vec<OutletId>,
}
impl<O> Default for Graph<O>
{
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
impl<O> Graph<O>
{
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
#[derive(Clone, Default)]
struct Outlet {
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
use std::ops::{Deref, DerefMut};
struct ModelPatch<O>
{
    pub model: Graph<O>,
    pub inputs: HashMap<usize, usize>,
    pub incoming: HashMap<OutletId, OutletId>,
    pub shunt_outlet_by: HashMap<OutletId, OutletId>,
}
impl<O> Default for ModelPatch<O>
{
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
            TractResult::Ok(vec!(OutletId::new(id, 0)))
        }
    }
}
use std::collections::HashMap;

fn dump_pfs(pfs: &ProtoFusedSpec) {
    /*
    let buf: &[u8;64] = unsafe { std::mem::transmute(pfs) };
    println!("{:?}", buf);a
    */
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
    let mm = model.wire_node(Box::new(MatMulUnary {}), &[source]).unwrap()[0];
    model.outputs = vec![mm];
    let patch = TypedModelPatch::replace_single_op(
        &model,
        &model.nodes[mm.node],
        &[source],
        Box::new(MatMulUnary {}),
    )
    .unwrap();
    patch.apply(&mut model).unwrap();

/*
    eprintln!("store:");
    dump_pfs(&ProtoFusedSpec::Store);
    eprintln!("bins:");
    dump_pfs(&ProtoFusedSpec::BinScalar(AttrOrInput::Input(4), BinOp::Min));
    dump_pfs(&ProtoFusedSpec::BinScalar(AttrOrInput::Input(4), BinOp::Max));
    dump_pfs(&ProtoFusedSpec::BinPerRow(AttrOrInput::Input(4), BinOp::Min));
    dump_pfs(&ProtoFusedSpec::BinPerCol(AttrOrInput::Input(4), BinOp::Min));
    eprintln!("add unicast (with input):");
    dump_pfs(&ProtoFusedSpec::AddUnicast(OutputStoreSpec::Strides { col_byte_stride: 3, mr: 3, nr: 3, m: 3, n: 3}, AttrOrInput::Input(2)));
    eprintln!("add unicast (with attr):");
    dump_pfs(&ProtoFusedSpec::AddUnicast(OutputStoreSpec::Strides { col_byte_stride: 3, mr: 3, nr: 3, m: 3, n: 3}, AttrOrInput::Attr(Arc::new(()))));
    eprintln!("add row col product:");
    dump_pfs(&ProtoFusedSpec::AddRowColProducts(AttrOrInput::Input(3), AttrOrInput::Input(4)));
*/

    let packed_as =
        Array::from_shape_fn(1, |_| (Box::new(()), vec!(ProtoFusedSpec::Store)));
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

fn main() {
    crasher_monterey()
}

#[test]
fn t1() {
	crasher_monterey()
}

#[test]
fn t2() {
    let mut buf: [u8;64] = [192, 129, 255, 162, 54, 38, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 168, 81, 97, 74, 158, 169, 127, 0, 8, 0, 0, 0, 0, 0, 0, 0, 108, 5, 70, 12, 248, 127, 0, 0, 32, 128, 121, 0, 0, 96, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let mut buf: [u8;64] = [192, 193, 202, 161, 57, 12, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 184, 113, 209, 169, 51, 191, 127, 0, 8, 0, 0, 0, 0, 0, 0, 0, 108, 5, 70, 12, 248, 127, 0, 0, 32, 192, 14, 3, 0, 96, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let pfs = buf.as_ptr() as *mut ProtoFusedSpec;
    unsafe { std::ptr::drop_in_place(pfs); }
}
