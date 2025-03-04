#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum State {
    #[default]
    Learning,
    Evaluating,
}