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
