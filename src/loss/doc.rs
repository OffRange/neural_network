#[macro_export]
macro_rules! doc_cross_entropy {
    (default $struct_name:ident $default:expr) => {
        concat!(
            "Creates a default instance of ", stringify!($struct_name), ".\n\n",
            "The default value for clamp_epsilon is set to ", stringify!($default), "."
        )
    };
    (new $struct_name:ident) => {
        concat!(
            "Creates a new instance of ", stringify!($struct_name), " with a specified clamp epsilon.\n\n",
            "# Arguments\n\n",
            "* `clamp_epsilon` - A small value to clamp predicted probabilities and avoid\n",
            "  numerical instability (e.g., taking the logarithm of zero).\n\n",
            "# Returns\n\n",
            "A new `", stringify!($struct_name), "` instance."
        )
    };
}