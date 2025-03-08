#[macro_export]
macro_rules! assert_eq_approx {
    ($left:expr, $right:expr) => {
        assert!(($left - $right).abs() < f64::EPSILON, "left: {}, right: {}", $left, $right);
    };

    ($left:expr, $right:expr, $($arg:tt)+) => {
        assert!(($left - $right).abs() < f64::EPSILON, $($arg)+);
    };
}

#[macro_export]
macro_rules! assert_arr_eq_approx {
    ($left:expr, $right:expr) => {
        for (&value, &expected) in $left.iter().zip($right.iter()) {
            $crate::assert_eq_approx!(value, expected);
        }
    };
}