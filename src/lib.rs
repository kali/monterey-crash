use std::marker::PhantomData;
use std::sync::Arc;
#[allow(non_snake_case)]
#[inline(always)]
pub fn Ix1(i0: Ix) -> Ix1 {
    Dim::new([i0])
}
pub type Ix0 = Dim<[Ix; 0]>;
pub type Ix1 = Dim<[Ix; 1]>;
pub type Ix2 = Dim<[Ix; 2]>;
pub type Ix3 = Dim<[Ix; 3]>;
pub type Ix4 = Dim<[Ix; 4]>;
pub type Ix5 = Dim<[Ix; 5]>;
pub type Ix6 = Dim<[Ix; 6]>;
pub type IxDyn = Dim<IxDynImpl>;
pub type ArrayView1<'a, A> = ArrayView<'a, A, Ix1>;
pub type ArrayViewMut1<'a, A> = ArrayViewMut<'a, A, Ix1>;
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
    ptr: NonNull<A>,
    len: usize,
    capacity: usize,
}
impl<A> OwnedRepr<A> {
    pub(crate) fn from(v: Vec<A>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let len = v.len();
        let capacity = v.capacity();
        let ptr = nonnull_from_vec_data(&mut v);
        Self { ptr, len, capacity }
    }
    pub(crate) fn as_slice(&self) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    pub(crate) fn as_ptr(&self) -> *const A {
        self.ptr.as_ptr()
    }
    pub(crate) fn as_nonnull_mut(&mut self) -> NonNull<A> {
        self.ptr
    }
    fn take_as_vec(&mut self) -> Vec<A> {
        let capacity = self.capacity;
        let len = self.len;
        self.len = 0;
        self.capacity = 0;
        unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), len, capacity) }
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
        if self.capacity > 0 {
            if !mem::needs_drop::<A>() {
                unimplemented!()
            }
            self.take_as_vec();
        }
    }
}
use std::mem::size_of;
use std::mem::MaybeUninit;
#[allow(clippy::missing_safety_doc)]
pub unsafe trait RawData: Sized {
    type Elem;
}
#[allow(clippy::missing_safety_doc)]
pub unsafe trait RawDataClone: RawData {
    unsafe fn clone_with_ptr(&self, ptr: NonNull<Self::Elem>) -> (Self, NonNull<Self::Elem>);
}
#[allow(clippy::missing_safety_doc)]
pub unsafe trait Data: RawData {
    #[allow(clippy::wrong_self_convention)]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
    where
        Self::Elem: Clone,
        D: Dimension;
    fn try_into_owned_nocopy<D>(
        self_: ArrayBase<Self, D>,
    ) -> Result<Array<Self::Elem, D>, ArrayBase<Self, D>>
    where
        D: Dimension;
    #[allow(clippy::wrong_self_convention)]
    fn to_shared<D>(self_: &ArrayBase<Self, D>) -> ArcArray<Self::Elem, D>
    where
        Self::Elem: Clone,
        D: Dimension,
    {
        unimplemented!()
    }
}
unsafe impl<A> RawData for OwnedArcRepr<A> {
    type Elem = A;
}
unsafe impl<A> RawData for OwnedRepr<A> {
    type Elem = A;
}
unsafe impl<A> Data for OwnedRepr<A> {
    #[inline]
    fn into_owned<D>(self_: ArrayBase<Self, D>) -> Array<Self::Elem, D>
    where
        A: Clone,
        D: Dimension,
    {
        unimplemented!()
    }
    #[inline]
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
unsafe impl<'a, A> RawData for ViewRepr<&'a A> {
    type Elem = A;
}
unsafe impl<'a, A> RawData for ViewRepr<&'a mut A> {
    type Elem = A;
}
#[allow(clippy::missing_safety_doc)]
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
pub struct ShapeError {}
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
    #[inline]
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
#[allow(clippy::missing_safety_doc)]
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
    pub(crate) strides: Strides<Contiguous>,
}
pub(crate) enum Contiguous {}
impl<D> Shape<D> {
    pub(crate) fn is_c(&self) -> bool {
        match self.strides {
            Strides::C => true,
            _ => false,
        }
    }
}
pub struct StrideShape<D> {
    pub(crate) dim: D,
    pub(crate) strides: Strides<D>,
}
pub(crate) enum Strides<D> {
    C,
    F,
    Custom(D),
}
impl<D> Strides<D> {
    pub(crate) fn strides_for_dim(self, dim: &D) -> D
    where
        D: Dimension,
    {
        match self {
            Strides::C => dim.default_strides(),
            Strides::F => dim.fortran_strides(),
            Strides::Custom(c) => {
                unimplemented!()
            }
        }
    }
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
        let st = if shape.is_c() {
            Strides::C
        } else {
            unimplemented!()
        };
        StrideShape {
            strides: st,
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
            strides: Strides::C,
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
pub trait DimMax<Other: Dimension> {
    type Output: Dimension;
}
impl<D: Dimension> DimMax<D> for D {
    type Output = D;
}
impl DimMax<Ix1> for Ix0 {
    type Output = Ix1;
}
impl DimMax<Ix0> for Ix1 {
    type Output = Ix1;
}
impl DimMax<Ix0> for Ix2 {
    type Output = Ix2;
}
impl DimMax<Ix0> for Ix3 {
    type Output = Ix3;
}
impl DimMax<Ix0> for Ix4 {
    type Output = Ix4;
}
impl DimMax<Ix0> for Ix5 {
    type Output = Ix5;
}
impl DimMax<Ix0> for Ix6 {
    type Output = Ix6;
}
impl DimMax<Ix2> for Ix1 {
    type Output = Ix2;
}
impl DimMax<Ix1> for Ix2 {
    type Output = Ix2;
}
impl DimMax<Ix3> for Ix2 {
    type Output = Ix3;
}
impl DimMax<Ix2> for Ix3 {
    type Output = Ix3;
}
impl DimMax<Ix4> for Ix3 {
    type Output = Ix4;
}
impl DimMax<Ix3> for Ix4 {
    type Output = Ix4;
}
impl DimMax<Ix5> for Ix4 {
    type Output = Ix5;
}
impl DimMax<Ix4> for Ix5 {
    type Output = Ix5;
}
impl DimMax<Ix6> for Ix5 {
    type Output = Ix6;
}
impl DimMax<Ix5> for Ix6 {
    type Output = Ix6;
}
impl DimMax<IxDyn> for Ix0 {
    type Output = IxDyn;
}
impl DimMax<Ix0> for IxDyn {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix1 {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix2 {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix3 {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix4 {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix5 {
    type Output = IxDyn;
}
impl DimMax<IxDyn> for Ix6 {
    type Output = IxDyn;
}
use num_traits::Zero;
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}
impl IntoDimension for Ix {
    type Dim = Ix1;
    #[inline(always)]
    fn into_dimension(self) -> Ix1 {
        unimplemented!()
    }
}
impl<D> IntoDimension for D
where
    D: Dimension,
{
    type Dim = D;
    #[inline(always)]
    fn into_dimension(self) -> Self {
        self
    }
}
impl IntoDimension for IxDynImpl {
    type Dim = IxDyn;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        Dim::new(self)
    }
}
impl IntoDimension for Vec<Ix> {
    type Dim = IxDyn;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        Dim::new(IxDynImpl::from(self))
    }
}
impl IntoDimension for () {
    type Dim = Dim<[Ix; 0]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix) {
    type Dim = Dim<[Ix; 2]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 3]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 4]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 5]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
impl IntoDimension for (Ix, Ix, Ix, Ix, Ix, Ix) {
    type Dim = Dim<[Ix; 6]>;
    #[inline(always)]
    fn into_dimension(self) -> Self::Dim {
        unimplemented!()
    }
}
pub struct Dim<I: ?Sized> {
    index: I,
}
#[automatically_derived]
impl<I: ::core::clone::Clone + ?Sized> ::core::clone::Clone for Dim<I> {
    #[inline]
    fn clone(&self) -> Dim<I> {
        Dim {
            index: ::core::clone::Clone::clone(&self.index),
        }
    }
}
#[automatically_derived]
impl<I: ::core::cmp::PartialEq + ?Sized> ::core::cmp::PartialEq for Dim<I> {
    #[inline]
    fn eq(&self, other: &Dim<I>) -> bool {
        unimplemented!()
    }
}
#[automatically_derived]
impl<I: ::core::cmp::Eq + ?Sized> ::core::cmp::Eq for Dim<I> {}
#[automatically_derived]
impl<I: ::core::default::Default + ?Sized> ::core::default::Default for Dim<I> {
    #[inline]
    fn default() -> Dim<I> {
        unimplemented!()
    }
}
impl<I> Dim<I> {
    pub(crate) fn new(index: I) -> Dim<I> {
        Dim { index }
    }
    #[inline(always)]
    pub(crate) fn ix(&self) -> &I {
        &self.index
    }
    #[inline(always)]
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
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
impl<I> Add for Dim<I>
where
    Dim<I>: Dimension,
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        unimplemented!()
    }
}
impl<I> AddAssign for Dim<I>
where
    Dim<I>: Dimension,
{
    fn add_assign(&mut self, rhs: Self) {
        unimplemented!()
    }
}
impl<'a, I> AddAssign<&'a Dim<I>> for Dim<I>
where
    Dim<I>: Dimension,
{
    fn add_assign(&mut self, rhs: &Self) {
        unimplemented!()
    }
}
impl<I> Sub for Dim<I>
where
    Dim<I>: Dimension,
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        unimplemented!()
    }
}
impl<I> SubAssign for Dim<I>
where
    Dim<I>: Dimension,
{
    fn sub_assign(&mut self, rhs: Self) {
        unimplemented!()
    }
}
impl<'a, I> SubAssign<&'a Dim<I>> for Dim<I>
where
    Dim<I>: Dimension,
{
    fn sub_assign(&mut self, rhs: &Self) {
        unimplemented!()
    }
}
impl<I> Mul for Dim<I>
where
    Dim<I>: Dimension,
{
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
        unimplemented!()
    }
}
impl<I> MulAssign for Dim<I>
where
    Dim<I>: Dimension,
{
    fn mul_assign(&mut self, rhs: Self) {
        unimplemented!()
    }
}
impl<'a, I> MulAssign<&'a Dim<I>> for Dim<I>
where
    Dim<I>: Dimension,
{
    fn mul_assign(&mut self, rhs: &Self) {
        unimplemented!()
    }
}
impl<I> Mul<Ix> for Dim<I>
where
    Dim<I>: Dimension,
{
    type Output = Self;
    fn mul(mut self, rhs: Ix) -> Self {
        unimplemented!()
    }
}
impl<I> MulAssign<Ix> for Dim<I>
where
    Dim<I>: Dimension,
{
    fn mul_assign(&mut self, rhs: Ix) {
        unimplemented!()
    }
}
pub trait Dimension:
    Clone
    + Eq
    + Send
    + Sync
    + Default
    + Add<Self, Output = Self>
    + AddAssign
    + for<'x> AddAssign<&'x Self>
    + Sub<Self, Output = Self>
    + SubAssign
    + for<'x> SubAssign<&'x Self>
    + Mul<usize, Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign
    + for<'x> MulAssign<&'x Self>
    + MulAssign<usize>
    + DimMax<Ix0, Output = Self>
    + DimMax<Self, Output = Self>
    + DimMax<IxDyn, Output = IxDyn>
    + DimMax<<Self as Dimension>::Smaller, Output = Self>
    + DimMax<<Self as Dimension>::Larger, Output = <Self as Dimension>::Larger>
    + DimAdd<Self>
    + DimAdd<<Self as Dimension>::Smaller>
    + DimAdd<<Self as Dimension>::Larger>
    + DimAdd<Ix0, Output = Self>
    + DimAdd<Ix1, Output = <Self as Dimension>::Larger>
    + DimAdd<IxDyn, Output = IxDyn>
{
    const NDIM: Option<usize>;
    type Pattern: IntoDimension<Dim = Self> + Clone + PartialEq + Eq + Default;
    type Smaller: Dimension;
    type Larger: Dimension;
    fn ndim(&self) -> usize;
    fn into_pattern(self) -> Self::Pattern;
    fn size(&self) -> usize {
        self.slice().iter().fold(1, |s, &a| s * a as usize)
    }
    fn size_checked(&self) -> Option<usize> {
        unimplemented!()
    }
    fn slice(&self) -> &[Ix];
    fn slice_mut(&mut self) -> &mut [Ix];
    fn as_array_view(&self) -> ArrayView1<'_, Ix> {
        unimplemented!()
    }
    fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, Ix> {
        unimplemented!()
    }
    fn equal(&self, rhs: &Self) -> bool {
        unimplemented!()
    }
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
    #[inline]
    fn first_index(&self) -> Option<Self> {
        unimplemented!()
    }
    #[inline]
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
    #[inline]
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
    type Smaller = Self;
    type Larger = Ix1;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 1]> {
    const NDIM: Option<usize> = Some(1);
    type Pattern = Ix;
    type Smaller = Ix0;
    type Larger = Ix2;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 2]> {
    const NDIM: Option<usize> = Some(2);
    type Pattern = (Ix, Ix);
    type Smaller = Ix1;
    type Larger = Ix3;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 3]> {
    const NDIM: Option<usize> = Some(3);
    type Pattern = (Ix, Ix, Ix);
    type Smaller = Ix2;
    type Larger = Ix4;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 4]> {
    const NDIM: Option<usize> = Some(4);
    type Pattern = (Ix, Ix, Ix, Ix);
    type Smaller = Dim<[Ix; 4 - 1]>;
    type Larger = Ix5;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 5]> {
    const NDIM: Option<usize> = Some(5);
    type Pattern = (Ix, Ix, Ix, Ix, Ix);
    type Smaller = Dim<[Ix; 5 - 1]>;
    type Larger = Ix6;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for Dim<[Ix; 6]> {
    const NDIM: Option<usize> = Some(6);
    type Pattern = (Ix, Ix, Ix, Ix, Ix, Ix);
    type Smaller = Dim<[Ix; 6 - 1]>;
    type Larger = IxDyn;
    #[inline]
    fn ndim(&self) -> usize {
        unimplemented!()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        unimplemented!()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        unimplemented!()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        unimplemented!()
    }
    #[inline]
    fn zeros(ndim: usize) -> Self {
        unimplemented!()
    }
}
impl Dimension for IxDyn {
    const NDIM: Option<usize> = None;
    type Pattern = Self;
    type Smaller = Self;
    type Larger = Self;
    #[inline]
    fn ndim(&self) -> usize {
        self.ix().len()
    }
    #[inline]
    fn slice(&self) -> &[Ix] {
        self.ix()
    }
    #[inline]
    fn slice_mut(&mut self) -> &mut [Ix] {
        self.ixm()
    }
    #[inline]
    fn into_pattern(self) -> Self::Pattern {
        self
    }
    #[inline]
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
    #[inline]
    fn clone(&self) -> IxDynImpl {
        IxDynImpl(::core::clone::Clone::clone(&self.0))
    }
}
#[automatically_derived]
impl ::core::cmp::PartialEq for IxDynImpl {
    #[inline]
    fn eq(&self, other: &IxDynImpl) -> bool {
        unimplemented!()
    }
}
#[automatically_derived]
impl ::core::cmp::Eq for IxDynImpl {}
#[automatically_derived]
impl ::core::default::Default for IxDynImpl {
    #[inline]
    fn default() -> IxDynImpl {
        unimplemented!()
    }
}
impl<'a> From<&'a [Ix]> for IxDynImpl {
    #[inline]
    fn from(ix: &'a [Ix]) -> Self {
        IxDynImpl(IxDynRepr::copy_from(ix))
    }
}
impl From<Vec<Ix>> for IxDynImpl {
    #[inline]
    fn from(ix: Vec<Ix>) -> Self {
        IxDynImpl(IxDynRepr::from_vec_auto(ix))
    }
}
impl Deref for IxDynImpl {
    type Target = [Ix];
    #[inline]
    fn deref(&self) -> &[Ix] {
        &self.0
    }
}
impl DerefMut for IxDynImpl {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Ix] {
        &mut self.0
    }
}
impl IxDyn {
    #[inline]
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
pub trait DimAdd<D: Dimension> {
    type Output: Dimension;
}
impl<D: Dimension> DimAdd<D> for Ix0 {
    type Output = D;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 1]> {
    type Output = Dim<[usize; 1 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 1]> {
    type Output = Dim<[usize; 1 + 1]>;
}
impl DimAdd<Dim<[usize; 2]>> for Dim<[usize; 1]> {
    type Output = Dim<[usize; 1 + 2]>;
}
impl DimAdd<IxDyn> for Dim<[usize; 1]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 2]> {
    type Output = Dim<[usize; 2 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 2]> {
    type Output = Dim<[usize; 2 + 1]>;
}
impl DimAdd<Dim<[usize; 2]>> for Dim<[usize; 2]> {
    type Output = Dim<[usize; 2 + 2]>;
}
impl DimAdd<Dim<[usize; 3]>> for Dim<[usize; 2]> {
    type Output = Dim<[usize; 2 + 3]>;
}
impl DimAdd<IxDyn> for Dim<[usize; 2]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 3]> {
    type Output = Dim<[usize; 3 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 3]> {
    type Output = Dim<[usize; 3 + 1]>;
}
impl DimAdd<Dim<[usize; 2]>> for Dim<[usize; 3]> {
    type Output = Dim<[usize; 3 + 2]>;
}
impl DimAdd<Dim<[usize; 3]>> for Dim<[usize; 3]> {
    type Output = Dim<[usize; 3 + 3]>;
}
impl DimAdd<Dim<[usize; 4]>> for Dim<[usize; 3]> {
    type Output = IxDyn;
}
impl DimAdd<IxDyn> for Dim<[usize; 3]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 4]> {
    type Output = Dim<[usize; 4 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 4]> {
    type Output = Dim<[usize; 4 + 1]>;
}
impl DimAdd<Dim<[usize; 3]>> for Dim<[usize; 4]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 4]>> for Dim<[usize; 4]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 5]>> for Dim<[usize; 4]> {
    type Output = IxDyn;
}
impl DimAdd<IxDyn> for Dim<[usize; 4]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 5]> {
    type Output = Dim<[usize; 5 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 5]> {
    type Output = Dim<[usize; 5 + 1]>;
}
impl DimAdd<Dim<[usize; 4]>> for Dim<[usize; 5]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 5]>> for Dim<[usize; 5]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 6]>> for Dim<[usize; 5]> {
    type Output = IxDyn;
}
impl DimAdd<IxDyn> for Dim<[usize; 5]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 0]>> for Dim<[usize; 6]> {
    type Output = Dim<[usize; 6 + 0]>;
}
impl DimAdd<Dim<[usize; 1]>> for Dim<[usize; 6]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 5]>> for Dim<[usize; 6]> {
    type Output = IxDyn;
}
impl DimAdd<Dim<[usize; 6]>> for Dim<[usize; 6]> {
    type Output = IxDyn;
}
impl DimAdd<IxDyn> for Dim<[usize; 6]> {
    type Output = IxDyn;
}
impl<D: Dimension> DimAdd<D> for IxDyn {
    type Output = IxDyn;
}
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
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
pub type ArcArray<A, D> = ArrayBase<OwnedArcRepr<A>, D>;
pub type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;
pub type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;
pub type ArrayViewMut<'a, A, D> = ArrayBase<ViewRepr<&'a mut A>, D>;
pub struct OwnedArcRepr<A>(Arc<OwnedRepr<A>>);
pub struct ViewRepr<A> {
    life: PhantomData<A>,
}
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
        if shape.is_c() {
            let v = to_vec_mapped(indices(shape.dim.clone()).into_iter(), f);
            unsafe { Self::from_shape_vec_unchecked(shape, v) }
        } else {
            unimplemented!()
        }
    }
    pub unsafe fn from_shape_vec_unchecked<Sh>(shape: Sh, v: Vec<A>) -> Self
    where
        Sh: Into<StrideShape<D>>,
    {
        let shape = shape.into();
        let dim = shape.dim;
        let strides = shape.strides.strides_for_dim(&dim);
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
