pub type Ix1 = Dim<[Ix; 1]>;
pub type IxDyn = Dim<IxDynImpl>;
use std::mem;
use std::ptr::NonNull;
#[repr(C)]
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
pub struct Axis(pub usize);
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}
impl IntoDimension for Ix {
    type Dim = Ix1;
    fn into_dimension(self) -> Ix1 {
        unimplemented!()
    }
}
impl IntoDimension for () {
    type Dim = Dim<[Ix; 0]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix) {
    type Dim = Dim<[Ix; 2]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 3]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 4]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 5]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 6]>;
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
pub struct Dim<I: ?Sized> {
    index: I,
}
#[automatically_derived]
impl<I: ::core::clone::Clone + ?Sized> ::core::clone::Clone for Dim<I> {
    fn clone(&self) -> Dim<I> {
        unimplemented!()
    }
}
#[automatically_derived]
impl<I: ::core::cmp::PartialEq + ?Sized> ::core::cmp::PartialEq for Dim<I> {
    fn eq(&self, other: &Dim<I>) -> bool {
        unimplemented!()
    }
}
#[automatically_derived]
impl<I: ::core::cmp::Eq + ?Sized> ::core::cmp::Eq for Dim<I> {}
#[automatically_derived]
impl<I: ::core::default::Default + ?Sized> ::core::default::Default for Dim<I> {
    fn default() -> Dim<I> {
        unimplemented!()
    }
}
pub trait Dimension: Clone + Eq + Send + Sync + Default {
    const NDIM: Option<usize>;
    type Pattern: IntoDimension<Dim = Self> + Clone + PartialEq + Eq + Default;
    fn ndim(&self) -> usize;
    fn into_pattern(self) -> Self::Pattern;
    fn size(&self) -> usize {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix];
    fn slice_mut(&mut self) -> &mut [Ix];
    fn default_strides(&self) -> Self {
        unimplemented!()
    }
    fn fortran_strides(&self) -> Self {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self;
    fn first_index(&self) -> Option<Self> {
        unimplemented!()
    }
    fn next_for(&self, index: Self) -> Option<Self> {
        unimplemented!()
    }
    fn next_for_f(&self, index: &mut Self) -> bool {
        unimplemented!()
    }
    fn strides_equivalent<D>(&self, strides1: &Self, strides2: &D) -> bool
    where
        D: Dimension,
    {
        unimplemented!()
    }
    fn stride_offset(index: &Self, strides: &Self) -> isize {
        unimplemented!()
    }
    fn stride_offset_checked(&self, strides: &Self, index: &Self) -> Option<isize> {
        unimplemented!()
    }
    fn last_elem(&self) -> usize {
        unimplemented!()
    }
    fn set_last_elem(&mut self, i: usize) {
        unimplemented!()
    }
    fn is_contiguous(dim: &Self, strides: &Self) -> bool {
        unimplemented!()
    }
    fn _fastest_varying_stride_order(&self) -> Self {
        unimplemented!()
    }
    fn min_stride_axis(&self, strides: &Self) -> Axis {
        unimplemented!()
    }
    fn max_stride_axis(&self, strides: &Self) -> Axis {
        unimplemented!()
    }
    fn into_dyn(self) -> IxDyn {
        unimplemented!()
    }
    fn from_dimension<D2: Dimension>(d: &D2) -> Option<Self> {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 0]> {
    const NDIM: Option<usize> = Some(0);
    type Pattern = ();
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 1]> {
    const NDIM: Option<usize> = Some(1);
    type Pattern = Ix;
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 2]> {
    const NDIM: Option<usize> = Some(2);
    type Pattern = (Ix, Ix);
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 3]> {
    const NDIM: Option<usize> = Some(3);
    type Pattern = (Ix, Ix, Ix);
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 4]> {
    const NDIM: Option<usize> = Some(4);
    type Pattern = (Ix, Ix, Ix, Ix);
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 5]> {
    const NDIM: Option<usize> = Some(5);
    type Pattern = (Ix, Ix, Ix, Ix, Ix);
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 6]> {
    const NDIM: Option<usize> = Some(6);
    type Pattern = (Ix, Ix, Ix, Ix, Ix, Ix);
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
const CAP: usize = 4;
enum IxDynRepr<T> {
    Inline(u32, [T; CAP]),
    Alloc(Box<[T]>),
}
pub struct IxDynImpl(IxDynRepr<Ix>);
pub type Ix = usize;
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
}
