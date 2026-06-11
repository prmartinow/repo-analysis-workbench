import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parsers.rust import parse_rust_file


class RustParserTest(unittest.TestCase):
    def test_extracts_modules_impls_methods_and_test_markers(self) -> None:
        source = """
use crate::inner::{Thing, Other};

/// API docs
pub mod api {
    /// Service docs
    pub struct Service;

    impl Service {
        /// Runs it
        pub async fn run(&self) {}
    }

    #[cfg(test)]
    mod tests {
        #[tokio::test]
        async fn smoke() {}
    }
}
"""
        parsed = parse_rust_file("src/lib.rs", source, "demo-crate", "demo_crate")

        self.assertEqual(len(parsed.imports), 1)
        self.assertEqual(parsed.imports[0].path, "crate::inner::{Thing, Other}")

        symbols_by_key = {(symbol.kind, symbol.name): symbol for symbol in parsed.symbols}

        api_module = symbols_by_key[("module", "api")]
        service = symbols_by_key[("struct", "Service")]
        service_impl = symbols_by_key[("impl", "impl Service")]
        run_method = symbols_by_key[("method", "run")]
        tests_module = symbols_by_key[("module", "tests")]
        smoke_test = symbols_by_key[("function", "smoke")]

        self.assertEqual(api_module.qualified_name, "demo_crate::api")
        self.assertEqual(service.container_local_id, api_module.local_id)
        self.assertEqual(service.docstring, "Service docs")
        self.assertEqual(service_impl.container_local_id, api_module.local_id)
        self.assertEqual(run_method.container_local_id, service_impl.local_id)
        self.assertEqual(run_method.docstring, "Runs it")
        self.assertTrue(tests_module.is_test)
        self.assertTrue(smoke_test.is_test)

    def test_tracks_struct_and_enum_container_spans(self) -> None:
        source = """
pub struct Service {
    pub helper: Helper,
    count: u64,
}

pub enum Mode {
    Idle,
    Active,
}
"""
        parsed = parse_rust_file("src/lib.rs", source, "demo-crate", "demo_crate")
        symbols_by_key = {(symbol.kind, symbol.name): symbol for symbol in parsed.symbols}

        service = symbols_by_key[("struct", "Service")]
        mode = symbols_by_key[("enum", "Mode")]

        self.assertEqual(service.span.start_line, 2)
        self.assertEqual(service.span.end_line, 5)
        self.assertEqual(mode.span.start_line, 7)
        self.assertEqual(mode.span.end_line, 10)

    def test_handles_multiline_impl_and_function_signatures(self) -> None:
        source = """
impl<T> Builder<T>
where
    T: Send,
{
    pub fn build(
        value: T,
    ) -> Self {
        Self(value)
    }
}
"""
        parsed = parse_rust_file("src/lib.rs", source, "demo-crate", "demo_crate")
        symbols_by_key = {(symbol.kind, symbol.name): symbol for symbol in parsed.symbols}

        impl_symbol = symbols_by_key[("impl", "impl Builder<T>")]
        build_method = symbols_by_key[("method", "build")]

        self.assertEqual(build_method.container_local_id, impl_symbol.local_id)
        self.assertIn("pub fn build", build_method.signature)
        self.assertEqual(build_method.span.start_line, 6)


if __name__ == "__main__":
    unittest.main()
