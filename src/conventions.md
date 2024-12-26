
# Names and Types
- The wrapper types List, Dict, Tuple, Fun, etc. are not to be used in signatures except when meant literally, as class names.
- Write `T | S` for `Union[T, S]`, and `T | None` for `Optional[T]`. `Union` and `Optional` themselves are not to be used in signatures.