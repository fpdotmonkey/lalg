use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Index, Mul};

use float_eq::float_eq;
use itertools::Itertools;

type Scalar = f64;

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(proptest_derive::Arbitrary))]
struct Matrix<const ROWS: usize, const COLS: usize> {
    data: [[Scalar; COLS]; ROWS],
}

macro_rules! matrix {
    ($( $( $x:expr ),*);* ) => {
        Matrix { data: [ $( [ $($x),* ] ),* ] }
    };
}

impl<const ROWS: usize, const COLS: usize> PartialEq for Matrix<ROWS, COLS> {
    /// Tests approximate equality elementwise
    fn eq(&self, other: &Matrix<ROWS, COLS>) -> bool {
        float_eq!(self.data, other.data, rmin_all <= 1.0e-10)
    }
}

/// General NxM matrix methods
impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    fn new(data: [[Scalar; COLS]; ROWS]) -> Self {
        Matrix { data }
    }

    fn rows(&self) -> Vec<Vector<COLS>> {
        self.data.iter().map(Vector::new).collect()
    }
    fn cols(&self) -> Vec<Vector<ROWS>> {
        (0..COLS)
            .map(|i| {
                Vector::new(
                    &self
                        .rows()
                        .iter()
                        .map(|row| row[i])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap(),
                )
            })
            .collect()
    }

    /// The identity matrix; a matrix of 1s on the diagonal and 0s else
    fn identity() -> Self {
        Self {
            data: (0..ROWS)
                .map(|i| {
                    let mut row = [0.0; COLS];
                    if i < row.len() {
                        row[i] = 1.0
                    }
                    row
                })
                .collect::<Vec<[Scalar; COLS]>>()
                .try_into()
                .unwrap(),
        }
    }

    /// The zero matrix; a matrix of all zeros
    fn zero() -> Self {
        Self {
            data: [[0.0; COLS]; ROWS],
        }
    }

    fn transpose(&self) -> Matrix<COLS, ROWS> {
        let mut data: [[Scalar; ROWS]; COLS] = [[0.0; ROWS]; COLS];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &elem) in row.iter().enumerate() {
                data[j][i] = elem;
            }
        }
        Matrix::<COLS, ROWS> { data }
    }
}

/// Square matrix methods
impl<const SIZE: usize> Matrix<SIZE, SIZE> {
    fn inverse(&self) -> Option<Self> {
        todo!()
    }
    fn determinant(&self) -> Scalar {
        if SIZE == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }
        todo!()
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<Scalar> for Matrix<ROWS, COLS> {
    type Output = Self;
    fn mul(self, s: Scalar) -> Self::Output {
        Self::Output {
            data: self
                .data
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|elem| elem * s)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Mul<Matrix<ROWS, COLS>> for Scalar {
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, m: Matrix<ROWS, COLS>) -> Self::Output {
        m * self
    }
}

impl<const ROWS: usize, const COLS: usize, const IN_COLS: usize> Mul<Matrix<IN_COLS, COLS>>
    for Matrix<ROWS, IN_COLS>
{
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, other: Matrix<IN_COLS, COLS>) -> Self::Output {
        Self::Output {
            data: self
                .rows()
                .iter()
                .map(|row| {
                    other
                        .cols()
                        .iter()
                        .map(|col| row.dot(col))
                        .collect::<Vec<_>>()
                        .try_into()
                        .expect("incompatible matrices")
                })
                .collect::<Vec<_>>()
                .try_into()
                .expect("wrong size output matrix"),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Add<Self> for Matrix<ROWS, COLS> {
    type Output = Self;

    fn add(self, other: Matrix<ROWS, COLS>) -> Self::Output {
        Self {
            data: self
                .data
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(j, elem)| elem + other.data[i][j])
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<const ROWS: usize, const COLS: usize> Display for Matrix<ROWS, COLS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "[")?;
        write!(
            f,
            "{}",
            self.data
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&e| format!("{:?}", e))
                        .intersperse(", ".into())
                        .collect::<String>()
                })
                .intersperse("; ".into())
                .collect::<String>()
        )?;
        write!(f, "]")
    }
}

impl<const ROWS: usize, const COLS: usize> Debug for Matrix<ROWS, COLS> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{self}")
    }
}

struct Vector<const N: usize> {
    elements: [Scalar; N],
}

impl<const N: usize> Vector<N> {
    fn new(elements: &[Scalar; N]) -> Self {
        Self {
            elements: *elements,
        }
    }

    fn dot(&self, other: &Self) -> Scalar {
        self.elements
            .iter()
            .enumerate()
            .map(|(i, elem)| elem * other.elements[i])
            .sum()
    }
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = Scalar;

    fn index(&self, i: usize) -> &Self::Output {
        &self.elements[i]
    }
}

fn main() {
    let a: Matrix<2, 3> = matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
    let b: Matrix<3, 2> = matrix![5.0, 4.0; 3.0, 2.0; 1.0, 1.0];
    let c: Matrix<2, 2> = a * b;
    let d: Matrix<3, 3> = b * a;
    println!("{c}");
    println!("{d}");
    println!("{}", Matrix::<2, 3>::identity());
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::TAU;

    /// Matrices with small values
    /// This is to avoid nonsense with 0*inf for tests that aren't
    /// concerned with that.
    fn small_matrix<const ROWS: usize, const COLS: usize>() -> BoxedStrategy<Matrix<ROWS, COLS>> {
        prop::array::uniform::<_, ROWS>(prop::array::uniform::<std::ops::Range<Scalar>, COLS>(
            -1.0e10..1.0e10,
        ))
        .prop_map(|a| Matrix::new(a))
        .boxed()
    }

    fn non_singular_matrix<const SIZE: usize>() -> BoxedStrategy<Matrix<SIZE, SIZE>> {
        small_matrix::<SIZE, SIZE>()
            .prop_filter("det(m) == 0 => singular", |&m| m.determinant() != 0.0)
            .boxed()
    }

    #[test]
    fn scalar_multiplication_special_cases() {
        assert_eq!(2.0 * Matrix::identity(), matrix![2.0, 0.0; 0.0, 2.0]);
        assert_eq!(
            -2.0 * matrix![1.0, 1.0; 1.0, 1.0],
            matrix![-2.0, -2.0; -2.0, -2.0]
        );
    }

    #[test]
    fn determinant_special_cases() {
        assert_eq!(Matrix::<2, 2>::zero().determinant(), 0.0);
        assert_eq!(Matrix::<2, 2>::identity().determinant(), 1.0);
        // rotation matrix
        let a = TAU / 6.0;
        assert_eq!(
            matrix![a.cos(), -a.sin(); a.sin(), a.cos()].determinant(),
            1.0
        );
        // scaling matrix
        assert_eq!(matrix![2.0, 0.0; 0.0, 2.0].determinant(), 4.0);
        // singular
        assert_eq!(
            matrix![1.0, 1.0, 0.0; 1.0, 1.0, 0.0; 1.0, 0.0, 1.0].determinant(),
            0.0
        );
        assert_eq!(
            matrix![1.0, 1.0, 0.0; 0.0, 0.0, 1.0; 1.0, 1.0, 1.0].determinant(),
            0.0
        );
        assert_eq!(
            matrix![0.0, 0.0, 0.0; 0.0, 1.0, 0.0; 1.0, 0.0, 1.0].determinant(),
            0.0
        );
    }

    #[test]
    fn inverse_special_cases() {
        assert_eq!(Matrix::<4, 4>::zero().inverse(), None);
        assert_eq!(
            Matrix::<4, 4>::identity().inverse(),
            Some(Matrix::<4, 4>::identity())
        );
        // rotation matrix
        let a = TAU / 6.0;
        assert_eq!(
            matrix![a.cos(), -a.sin(); a.sin(), a.cos()].inverse(),
            Some(matrix![-a.cos(), a.sin(); -a.sin(), -a.cos()])
        );
        // scaling matrix
        assert_eq!(
            matrix![2.0, 0.0; 0.0, 2.0].inverse(),
            Some(matrix![0.5, 0.0; 0.0, 0.5])
        );
        // singular
        assert_eq!(
            matrix![1.0, 1.0, 0.0; 1.0, 1.0, 0.0; 1.0, 0.0, 1.0].inverse(),
            None
        );
        assert_eq!(
            matrix![1.0, 1.0, 0.0; 0.0, 0.0, 1.0; 1.0, 1.0, 1.0].inverse(),
            None
        );
        assert_eq!(
            matrix![0.0, 0.0, 0.0; 0.0, 1.0, 0.0; 1.0, 0.0, 1.0].inverse(),
            None
        );
    }

    proptest! {
        #[test]
        fn scalar_multiplication(
            a in small_matrix::<4, 3>(),
            b in small_matrix::<3, 4>(),
            scalar in -1.0e10..1.0e10 // ranged to avoid 0*inf nonsense
        ) {
            prop_assert_eq!(scalar * a, a * scalar);
            prop_assert_eq!(scalar * (a * b), (scalar * a) * b);
            prop_assert_eq!((a * b) * scalar, a * (b * scalar));
            prop_assert_eq!(scalar * Matrix::<4, 2>::zero(), Matrix::zero());
        }

        #[test]
        fn matrix_addition(
            a in small_matrix::<4, 4>(),
            b in small_matrix::<4, 4>(),
            c in small_matrix::<4, 4>()
        ) {
            // additive identity
            prop_assert_eq!(a + Matrix::zero(), a);
            // commutative
            prop_assert_eq!(a + b, b + a);
            // associative
            prop_assert_eq!((a + b) + c, a + (b + c));
        }

        #[test]
        fn matrix_multiplaction(
            a in small_matrix::<4, 4>(),
            b in small_matrix::<4, 4>(),
            c in small_matrix::<4, 4>(),
            d in small_matrix::<4, 3>(),
            e in small_matrix::<3, 4>()
        ) {
            // multiplicative identity
            prop_assert_eq!(a * Matrix::identity(), a);
            prop_assert_eq!(Matrix::identity() * a, a);
            prop_assert_eq!(d * Matrix::identity(), d);
            prop_assert_eq!(Matrix::identity() * d, d);
            // zero
            prop_assert_eq!(a * Matrix::<4, 4>::zero(), Matrix::zero());
            prop_assert_eq!(Matrix::<4, 4>::zero() * a, Matrix::zero());
            prop_assert_eq!(d * Matrix::<3, 3>::zero(), Matrix::zero());
            prop_assert_eq!(Matrix::<4, 4>::zero() * d, Matrix::zero());
            // associative
            prop_assert_eq!((a * b) * c, a * (b * c));
            prop_assert_eq!((a * d) * e, a * (d * e));
            // distributive
            prop_assert_eq!(a * (b + c), a * b + a * c);
            prop_assert_eq!((a + b) * c, a * c + b * c);
            prop_assert_eq!(e * (b + c), e * b + e * c);
            prop_assert_eq!((a + b) * d, a * d + b * d);
        }

        #[test]
        fn transpose(
            a in small_matrix::<4, 4>(),
            b in small_matrix::<4, 4>()
        ) {
            prop_assert_eq!(a.transpose().transpose(), a);
            prop_assert_eq!((a + b).transpose(), a.transpose() + b.transpose());
            prop_assert_eq!((a * b).transpose(), b.transpose() * a.transpose());
        }

        #[test]
        fn determinant(
            a in small_matrix::<4, 4>(),
            b in small_matrix::<4, 4>(),
            c in small_matrix::<4, 4>(),
            scalar in -1.0e10..1.0e10
        ) {
            let scalar: Scalar = scalar;
            prop_assert_eq!((a * b).determinant(), (b * a).determinant());
            prop_assert_eq!((a * b).determinant(), a.determinant() * b.determinant());
            // determinant of scalar multiplication gains scalar**dimension
            prop_assert_eq!((scalar * a).determinant(), scalar.powf(4.0) * a.determinant());
            // determinant of transpose is the same as matrix
            prop_assert_eq!(a.transpose().determinant(), a.determinant());
            // determinant of sums
            prop_assert!(
                (a + b + c).determinant() + c.determinant()
                >= (a + c).determinant() + (b + c).determinant()
            );
            prop_assert!(
                (a + b).determinant() >= a.determinant() + b.determinant()
            );
        }

        #[test]
        fn inverse(
            a in non_singular_matrix::<4>(),
            b in non_singular_matrix::<4>(),
            c in non_singular_matrix::<4>()
        ) {
            // multiplicative inverse
            prop_assert_eq!(a.inverse().unwrap() * a, Matrix::identity());
            prop_assert_eq!(a * a.inverse().unwrap(), Matrix::identity());
            // inverse of inverse of matrix is matrix
            prop_assert_eq!(a.inverse().unwrap().inverse().unwrap(), a);
            // inverse of matrix multiplication is matrix multiplication
            // of inverses in reverse order
            prop_assert_eq!(
                (a * b).inverse().unwrap(),
                b.inverse().unwrap() * a.inverse().unwrap()
            );
            prop_assert_eq!(
                (a * b * c).inverse().unwrap(),
                c.inverse().unwrap() * b.inverse().unwrap() * a.inverse().unwrap()
            );
            // determinant of inverse is inverse of determinant
            prop_assert_eq!(a.inverse().unwrap().determinant(), 1.0 / a.determinant());
        }
    }
}
