use crate::{call_helper, Helper};

pub struct Helper;

pub fn call_helper(_helper: Helper) {}

fn make_count() -> u64 {
    1
}

pub struct Service {
    pub helper: Helper,
    count: u64,
}

pub enum Mode {
    Idle,
    Active,
}

impl Service {
    pub fn build(helper: Helper) -> Self {
        let count = make_count();
        call_helper(helper);
        Self { helper: Helper, count }
    }
}
