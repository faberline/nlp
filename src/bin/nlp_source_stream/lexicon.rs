#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorEntry {
    surface: String,
    canonical: String,
}

/// Caller-supplied work vocabulary. Omitted categories are empty.
#[derive(serde::Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct WorkFile {
    actors: Vec<ActorEntry>,
    objects: Vec<ActorEntry>,
    ranks: Vec<ActorEntry>,
    places: Vec<String>,
    formed_ranks: Vec<String>,
    rank_blocked_prefixes: Vec<String>,
    footers: Vec<String>,
    line_footers: Vec<String>,
}

pub(super) struct WorkRefs<'a> {
    actors: Vec<(&'a str, &'a str)>,
    objects: Vec<(&'a str, &'a str)>,
    ranks: Vec<(&'a str, &'a str)>,
    places: Vec<&'a str>,
    formed_ranks: Vec<&'a str>,
    rank_blocked_prefixes: Vec<&'a str>,
    footers: Vec<&'a str>,
    line_footers: Vec<&'a str>,
}

impl WorkFile {
    pub fn parse(bytes: &[u8]) -> anyhow::Result<Self> {
        use anyhow::ensure;
        let file: Self = serde_json::from_slice(bytes)?;
        for (name, entries) in [
            ("actors", &file.actors),
            ("objects", &file.objects),
            ("ranks", &file.ranks),
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for entry in entries {
                ensure!(
                    !entry.surface.is_empty() && entry.surface.trim() == entry.surface,
                    "{name} surface must be nonempty and have no outer whitespace"
                );
                ensure!(
                    !entry.canonical.is_empty() && entry.canonical.trim() == entry.canonical,
                    "{name} canonical must be nonempty and have no outer whitespace"
                );
                ensure!(seen.insert(&entry.surface), "duplicate {name} surface");
            }
        }
        for (name, entries) in [
            ("places", &file.places),
            ("formed_ranks", &file.formed_ranks),
            ("rank_blocked_prefixes", &file.rank_blocked_prefixes),
            ("footers", &file.footers),
            ("line_footers", &file.line_footers),
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for entry in entries {
                ensure!(
                    !entry.is_empty() && entry.trim() == entry,
                    "{name} entries must be nonempty and have no outer whitespace"
                );
                ensure!(seen.insert(entry), "duplicate {name} entry");
            }
        }
        let rank_surfaces: std::collections::BTreeSet<&str> = file
            .ranks
            .iter()
            .map(|entry| entry.surface.as_str())
            .collect();
        ensure!(
            file.formed_ranks
                .iter()
                .all(|rank| rank_surfaces.contains(rank.as_str())),
            "each formed_ranks entry must match a ranks surface"
        );
        Ok(file)
    }
    pub fn refs(&self) -> WorkRefs<'_> {
        fn pairs(entries: &[ActorEntry]) -> Vec<(&str, &str)> {
            entries
                .iter()
                .map(|entry| (entry.surface.as_str(), entry.canonical.as_str()))
                .collect()
        }
        fn strings(entries: &[String]) -> Vec<&str> {
            entries.iter().map(String::as_str).collect()
        }
        WorkRefs {
            actors: pairs(&self.actors),
            objects: pairs(&self.objects),
            ranks: pairs(&self.ranks),
            places: strings(&self.places),
            formed_ranks: strings(&self.formed_ranks),
            rank_blocked_prefixes: strings(&self.rank_blocked_prefixes),
            footers: strings(&self.footers),
            line_footers: strings(&self.line_footers),
        }
    }
}

impl WorkRefs<'_> {
    pub fn lexicon(&self) -> faberline_nlp::Lexicon<'_> {
        faberline_nlp::Lexicon {
            actors: &self.actors,
            objects: &self.objects,
            ranks: &self.ranks,
            places: &self.places,
            formed_ranks: &self.formed_ranks,
            rank_blocked_prefixes: &self.rank_blocked_prefixes,
            footers: &self.footers,
            line_footers: &self.line_footers,
        }
    }
}
