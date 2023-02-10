#[allow(non_snake_case)]
pub fn Ix1(i0: Ix) -> Ix1 {
    Dim::new([i0])
}
pub type Ix0 = Dim<[Ix; 0]>;
pub type Ix1 = Dim<[Ix; 1]>;
pub type IxDyn = Dim<IxDynImpl>;
pub(crate) fn zip<I, J>(i: I, j: J) -> std::iter::Zip<I::IntoIter, J::IntoIter>
where
    I: IntoIterator,
    J: IntoIterator,
{
    i.into_iter().zip(j)
}
use std::mem;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
#[repr(C)]
pub struct OwnedRepr<A> {
     it: Vec<A>
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
    fn take_as_vec(&mut self) -> Vec<A> {
        std::mem::take(&mut self.it)
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
impl<A> Drop for OwnedRepr<A> {
    fn drop(&mut self) {
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
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
    where
        Self::Elem: Clone,
        D: Dimension;
    fn try_into_owned_nocopy<D>(
        self_: ArrayBase<Self, D>,
    ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
    where
        D: Dimension;
}
unsafe impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
}
unsafe impl<A> Data for OwnedRepr<A> {
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
    where
        A: Clone,
        D: Dimension,
    {
        unimplemented!()
    }
    fn try_into_owned_nocopy<D>(
        self_: ArrayBase<Self, D>,
    ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
    where
        D: Dimension,
    {
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
pub(crate) fn nonnull_from_vec_data<T>(v: &mut Vec<T>) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(v.as_mut_ptr()) }
}
pub struct IndicesIter<D> {
    dim: D,
    index: Option<D>,
}
pub fn indices<E>(shape: E) -> Indices<E::Dim>
where
    E: IntoDimension,
{
    let dim = shape.into_dimension();
    Indices {
        start: E::Dim::zeros(dim.ndim()),
        dim,
    }
}
impl<D> Iterator for IndicesIter<D>
where
    D: Dimension,
{
    type Item = D::Pattern;
    fn next(&mut self) -> Option<Self::Item> {
        let index = match self.index {
            None => return None,
            Some(ref ix) => ix.clone(),
        };
        self.index = self.dim.next_for(index.clone());
        Some(index.into_pattern())
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = match self.index {
            None => 0,
            Some(ref ix) => {
                let gone = self
                    .dim
                    .default_strides()
                    .slice()
                    .iter()
                    .zip(ix.slice().iter())
                    .fold(0, |s, (&a, &b)| s + a as usize * b as usize);
                self.dim.size() - gone
            }
        };
        (l, Some(l))
    }
}
impl<D> ExactSizeIterator for IndicesIter<D> where D: Dimension {}
impl<D> IntoIterator for Indices<D>
where
    D: Dimension,
{
    type Item = D::Pattern;
    type IntoIter = IndicesIter<D>;
    fn into_iter(self) -> Self::IntoIter {
        let sz = self.dim.size();
        let index = if sz != 0 {
            Some(self.start)
        } else {
            unimplemented!()
        };
        IndicesIter {
            index,
            dim: self.dim,
        }
    }
}
pub struct Indices<D>
where
    D: Dimension,
{
    start: D,
    dim: D,
}
use std::ptr;
pub unsafe trait TrustedIterator {}
unsafe impl<D> TrustedIterator for IndicesIter<D> where D: Dimension {}
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
pub struct Shape<D> {
    pub(crate) dim: D,
}
pub struct StrideShape<D> {
    pub(crate) dim: D,
}
pub trait ShapeBuilder {
    type Dim: Dimension;
    type Strides;
    fn into_shape(self) -> Shape<Self::Dim>;
}
impl<T, D> From<T> for StrideShape<D>
where
    D: Dimension,
    T: ShapeBuilder<Dim = D>,
{
    fn from(value: T) -> Self {
        let shape = value.into_shape();
        StrideShape {
            dim: shape.dim,
        }
    }
}
impl<T> ShapeBuilder for T
where
    T: IntoDimension,
{
    type Dim = T::Dim;
    type Strides = T;
    fn into_shape(self) -> Shape<Self::Dim> {
        Shape {
            dim: self.into_dimension(),
        }
    }
}
impl<D> ShapeBuilder for Shape<D>
where
    D: Dimension,
{
    type Dim = D;
    type Strides = D;
    fn into_shape(self) -> Shape<D> {
        self
    }
}
pub struct Axis(pub usize);
use num_traits::Zero;
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
impl<D> IntoDimension for D
where
    D: Dimension,
{
    type Dim = D;
    fn into_dimension(self) -> Self {
        self
    }
}
impl IntoDimension for IxDynImpl {
    type Dim = IxDyn;
    fn into_dimension(self) -> Self::Dim {
        Dim::new(self)
    }
}
impl IntoDimension for Vec<Ix> {
    type Dim = IxDyn;
    fn into_dimension(self) -> Self::Dim {
        Dim::new(IxDynImpl::from(self))
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
        Dim {
            index: ::core::clone::Clone::clone(&self.index),
        }
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
impl<I> Dim<I> {
    pub(crate) fn new(index: I) -> Dim<I> {
        Dim { index }
    }
    pub(crate) fn ix(&self) -> &I {
        &self.index
    }
    pub(crate) fn ixm(&mut self) -> &mut I {
        &mut self.index
    }
}
#[allow(non_snake_case)]
pub fn Dim<T>(index: T) -> T::Dim
where
    T: IntoDimension,
{
    index.into_dimension()
}
pub trait Dimension: Clone + Eq + Send + Sync + Default {
    const NDIM: Option<usize>;
    type Pattern: IntoDimension<Dim = Self> + Clone + PartialEq + Eq + Default;
    fn ndim(&self) -> usize;
    fn into_pattern(self) -> Self::Pattern;
    fn size(&self) -> usize {
        self.slice().iter().fold(1, |s, &a| s * a as usize)
    }
    fn slice(&self) -> &[Ix];
    fn slice_mut(&mut self) -> &mut [Ix];
    fn default_strides(&self) -> Self {
        let mut strides = Self::zeros(self.ndim());
        if self.slice().iter().all(|&d| d != 0) {
            let mut it = strides.slice_mut().iter_mut().rev();
            if let Some(rs) = it.next() {
                *rs = 1;
            }
            let mut cum_prod = 1;
            for (rs, dim) in it.zip(self.slice().iter().rev()) {
                cum_prod *= *dim;
                *rs = cum_prod;
            }
        }
        strides
    }
    fn fortran_strides(&self) -> Self {
        unimplemented!()
    }
    fn zeros(ndim: usize) -> Self;
    fn first_index(&self) -> Option<Self> {
        unimplemented!()
    }
    fn next_for(&self, index: Self) -> Option<Self> {
        let mut index = index;
        let mut done = false;
        for (&dim, ix) in zip(self.slice(), index.slice_mut()).rev() {
            *ix += 1;
            if *ix == dim {
                *ix = 0;
            } else {
                unimplemented!()
            }
        }
        if done {
            unimplemented!()
        } else {
            None
        }
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
impl Dimension for IxDyn {
    const NDIM: Option<usize> = None;
    type Pattern = Self;
    fn ndim(&self) -> usize {
        self.ix().len()
    }
    fn slice(&self) -> &[Ix] {
        self.ix()
    }
    fn slice_mut(&mut self) -> &mut [Ix] {
        self.ixm()
    }
    fn into_pattern(self) -> Self::Pattern {
        self
    }
    fn zeros(ndim: usize) -> Self {
        IxDyn::zeros(ndim)
    }
}
use std::ops::{Deref, DerefMut};
const CAP: usize = 4;
enum IxDynRepr<T> {
    Inline(u32, [T; CAP]),
    Alloc(Box<[T]>),
}
impl<T> Deref for IxDynRepr<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match *self {
            IxDynRepr::Inline(len, ref ar) => unsafe { ar.get_unchecked(..len as usize) },
            IxDynRepr::Alloc(ref ar) => ar,
        }
    }
}
impl<T> DerefMut for IxDynRepr<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match *self {
            IxDynRepr::Inline(len, ref mut ar) => unsafe { ar.get_unchecked_mut(..len as usize) },
            IxDynRepr::Alloc(ref mut ar) => ar,
        }
    }
}
impl<T: Copy + Zero> IxDynRepr<T> {
    pub fn copy_from(x: &[T]) -> Self {
        if x.len() <= CAP {
            let mut arr = [T::zero(); CAP];
            arr[..x.len()].copy_from_slice(x);
            IxDynRepr::Inline(x.len() as _, arr)
        } else {
            unimplemented!()
        }
    }
}
impl<T: Copy + Zero> IxDynRepr<T> {
    fn from_vec_auto(v: Vec<T>) -> Self {
        if v.len() <= CAP {
            Self::copy_from(&v)
        } else {
            unimplemented!()
        }
    }
}
impl<T: Copy> IxDynRepr<T> {
    fn from(x: &[T]) -> Self {
        unimplemented!()
    }
}
impl<T: Copy> Clone for IxDynRepr<T> {
    fn clone(&self) -> Self {
        match *self {
            IxDynRepr::Inline(len, arr) => IxDynRepr::Inline(len, arr),
            _ => Self::from(&self[..]),
        }
    }
}
pub struct IxDynImpl(IxDynRepr<Ix>);
#[automatically_derived]
impl ::core::clone::Clone for IxDynImpl {
    fn clone(&self) -> IxDynImpl {
        IxDynImpl(::core::clone::Clone::clone(&self.0))
    }
}
#[automatically_derived]
impl ::core::cmp::PartialEq for IxDynImpl {
    fn eq(&self, other: &IxDynImpl) -> bool {
        unimplemented!()
    }
}
#[automatically_derived]
impl ::core::cmp::Eq for IxDynImpl {}
#[automatically_derived]
impl ::core::default::Default for IxDynImpl {
    fn default() -> IxDynImpl {
        unimplemented!()
    }
}
impl<'a> From<&'a [Ix]> for IxDynImpl {
    fn from(ix: &'a [Ix]) -> Self {
        IxDynImpl(IxDynRepr::copy_from(ix))
    }
}
impl From<Vec<Ix>> for IxDynImpl {
    fn from(ix: Vec<Ix>) -> Self {
        IxDynImpl(IxDynRepr::from_vec_auto(ix))
    }
}
impl Deref for IxDynImpl {
    type Target = [Ix];
    fn deref(&self) -> &[Ix] {
        &self.0
    }
}
impl DerefMut for IxDynImpl {
    fn deref_mut(&mut self) -> &mut [Ix] {
        &mut self.0
    }
}
impl IxDyn {
    pub fn zeros(n: usize) -> IxDyn {
        const ZEROS: &[usize] = &[0; 4];
        if n <= ZEROS.len() {
            Dim(&ZEROS[..n])
        } else {
            unimplemented!()
        }
    }
}
impl<'a> IntoDimension for &'a [Ix] {
    type Dim = IxDyn;
    fn into_dimension(self) -> Self::Dim {
        Dim(IxDynImpl::from(self))
    }
}
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ()> {
    let size_nonzero = dim
        .slice()
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .unwrap();
    if size_nonzero > ::std::isize::MAX as usize {
        unimplemented!()
    } else {
        Ok(dim.size())
    }
}
pub fn offset_from_low_addr_ptr_to_logical_ptr<D: Dimension>(dim: &D, strides: &D) -> usize {
    let offset = IntoIterator::into_iter(dim.slice())
        .zip(strides.slice())
        .fold(0, |_offset, (&d, &s)| {
            let s = s as isize;
            if s < 0 && d > 1 {
                unimplemented!()
            } else {
                _offset
            }
        });
    offset as usize
}
pub type Ix = usize;
pub struct ArrayBase<S, D>
where
    S: RawData,
{
    data: S,
    ptr: std::ptr::NonNull<S::Elem>,
    dim: D,
    strides: D,
}
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;
impl<S: RawDataClone, D: Clone> Clone for ArrayBase<S, D> {
    fn clone(&self) -> ArrayBase<S, D> {
        unsafe {
            let (data, ptr) = self.data.clone_with_ptr(self.ptr);
            ArrayBase {
                data,
                ptr,
                dim: self.dim.clone(),
                strides: self.strides.clone(),
            }
        }
    }
}
impl<A, S> ArrayBase<S, Ix1>
where
    S: RawData<Elem = A>,
{
    pub(crate) unsafe fn from_data_ptr(data: S, ptr: NonNull<A>) -> Self {
        let array = ArrayBase {
            data,
            ptr,
            dim: Ix1(0),
            strides: Ix1(1),
        };
        array
    }
}
impl<A, S, D> ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    pub(crate) unsafe fn with_strides_dim<E>(self, strides: E, dim: E) -> ArrayBase<S, E>
    where
        E: Dimension,
    {
        ArrayBase {
            data: self.data,
            ptr: self.ptr,
            dim,
            strides,
        }
    }
}
impl<S, A, D> ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    pub fn from_shape_fn<Sh, F>(shape: Sh, f: F) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
    {
        let shape = shape.into_shape();
        let _ = match size_of_shape_checked(&shape.dim) {
            Ok(sz) => sz,
            Err(_) => {
                unimplemented!()
            }
        };
            let v = to_vec_mapped(indices(shape.dim.clone()).into_iter(), f);
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
    }
    pub unsafe fn from_shape_vec_unchecked<Sh>(shape: Sh, v: Vec<A>) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = dim.default_strides();
        Self::from_vec_dim_stride_unchecked(dim, strides, v)
    }
    unsafe fn from_vec_dim_stride_unchecked(dim: D, strides: D, mut v: Vec<A>) -> Self {
        let ptr = std::ptr::NonNull::new(
            v.as_mut_ptr()
                .add(offset_from_low_addr_ptr_to_logical_ptr(&dim, &strides)),
        )
        .unwrap();
        ArrayBase::from_data_ptr(DataOwned::new(v), ptr).with_strides_dim(strides, dim)
    }
}
