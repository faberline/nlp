// Character offsets count Unicode scalar values, not bytes or grapheme clusters.
// A Chapter builds this from its source once and never changes that source.
const STRIDE: usize = 256;

// Most chapters fit in 32-bit character offsets. Keep a wide fallback so the
// representation never truncates positions in unusually large inputs.
enum Offsets {
    Compact(Vec<u32>),
    Wide(Vec<usize>),
}

impl Offsets {
    fn new() -> Self {
        Self::Compact(Vec::new())
    }

    fn push(&mut self, offset: usize) {
        match self {
            Self::Compact(offsets) => match u32::try_from(offset) {
                Ok(value) => offsets.push(value),
                Err(_) => {
                    let mut wide: Vec<usize> =
                        offsets.iter().map(|&value| value as usize).collect();
                    wide.push(offset);
                    *self = Self::Wide(wide);
                }
            },
            Self::Wide(offsets) => offsets.push(offset),
        }
    }

    #[cfg(test)]
    fn count_before(&self, offset: usize) -> usize {
        match self {
            Self::Compact(offsets) => offsets.partition_point(|&at| (at as usize) < offset),
            Self::Wide(offsets) => offsets.partition_point(|&at| at < offset),
        }
    }

    fn get(&self, index: usize) -> usize {
        match self {
            Self::Compact(offsets) => offsets[index] as usize,
            Self::Wide(offsets) => offsets[index],
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::Compact(offsets) => offsets.len(),
            Self::Wide(offsets) => offsets.len(),
        }
    }
}

#[cfg(test)]
pub(crate) struct LineBreaks(Offsets);

#[cfg(test)]
impl LineBreaks {
    pub(crate) fn new() -> Self {
        Self(Offsets::new())
    }
    pub(crate) fn push(&mut self, offset: usize) {
        self.0.push(offset);
    }
    pub(crate) fn line(&self, offset: usize) -> usize {
        self.0.count_before(offset) + 1
    }
}

pub(crate) struct SparseOffsets {
    checkpoints: Offsets,
    #[cfg(test)]
    scanned: std::cell::Cell<usize>,
}

impl SparseOffsets {
    pub(crate) fn new(source: &str) -> Self {
        let mut checkpoints = Offsets::new();
        for byte in source
            .char_indices()
            .map(|(byte, _)| byte)
            .chain(std::iter::once(source.len()))
            .enumerate()
            .filter_map(|(index, byte)| (index % STRIDE == 0).then_some(byte))
        {
            checkpoints.push(byte);
        }
        Self {
            checkpoints,
            #[cfg(test)]
            scanned: std::cell::Cell::new(0),
        }
    }

    // The caller validates the character offset against the original source first.
    pub(crate) fn byte_offset(&self, source: &str, character: usize) -> usize {
        let mut byte = self.checkpoints.get(character / STRIDE);
        #[cfg(test)]
        self.scanned.set(0);
        for ch in source[byte..].chars().take(character % STRIDE) {
            byte += ch.len_utf8();
            #[cfg(test)]
            self.scanned.set(self.scanned.get() + 1);
        }
        byte
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_and_wide_line_offsets_match_the_oracle() {
        let mut lines = LineBreaks::new();
        for offset in [1, 4, 100] {
            lines.push(offset);
        }
        for at in 0..=102 {
            assert_eq!(
                lines.line(at),
                1 + [1, 4, 100].iter().filter(|&&n| n < at).count()
            );
        }
        assert!(matches!(lines.0, Offsets::Compact(_)));
        #[cfg(target_pointer_width = "64")]
        {
            let boundary = u32::MAX as usize;
            lines.push(boundary);
            lines.push(boundary + 1);
            assert!(matches!(lines.0, Offsets::Wide(_)));
            assert_eq!(lines.line(boundary), 4);
            assert_eq!(lines.line(boundary + 1), 5);
            assert_eq!(lines.line(boundary + 2), 6);
            let mut bytes = Offsets::new();
            bytes.push(0);
            bytes.push(boundary);
            bytes.push(boundary + 1);
            assert!(matches!(bytes, Offsets::Wide(_)));
            assert_eq!(
                (bytes.get(0), bytes.get(1), bytes.get(2)),
                (0, boundary, boundary + 1)
            );
        }
    }

    #[test]
    fn all_ranges_match_scalar_oracle_across_checkpoints() {
        let source = "A𠮷e\u{301}\r\n中".repeat(45);
        let index = SparseOffsets::new(&source);
        let count = source.chars().count();
        for start in 0..=count {
            for end in start..=count {
                let start_byte = index.byte_offset(&source, start);
                assert!(index.scanned.get() < STRIDE);
                let end_byte = index.byte_offset(&source, end);
                assert!(index.scanned.get() < STRIDE);
                let expected: String = source.chars().skip(start).take(end - start).collect();
                assert_eq!(&source[start_byte..end_byte], expected);
            }
        }
        assert_eq!(index.checkpoints.len(), count / STRIDE + 1);
    }

    #[test]
    fn late_offsets_have_a_fixed_scan_bound_and_handle_exact_end() {
        for count in [0, STRIDE - 1, STRIDE, STRIDE + 1, 100_000] {
            let source: String = "𠮷e\u{301}\r\n中".chars().cycle().take(count).collect();
            let index = SparseOffsets::new(&source);
            assert_eq!(index.checkpoints.len(), count / STRIDE + 1);
            for at in count.saturating_sub(300)..=count {
                let byte = index.byte_offset(&source, at);
                assert_eq!(index.scanned.get(), at % STRIDE);
                assert!(index.scanned.get() < STRIDE);
                assert_eq!(source[..byte].chars().count(), at);
            }
        }
    }
}
