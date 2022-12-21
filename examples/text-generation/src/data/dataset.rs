use burn::data::dataset::{
    source::huggingface::downloader::HuggingfaceDatasetLoader, Dataset, InMemDataset,
};

#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct C4Item {
    pub text: String,
}

pub struct C4Dataset {
    dataset: InMemDataset<C4Item>,
}

impl Dataset<TextGenerationItem> for C4Dataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.text))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl C4Dataset {
    pub fn train() -> Self {
        let dataset: InMemDataset<C4Item> = HuggingfaceDatasetLoader::new("c4", "train")
            .extract_string("text")
            .config("en")
            .load_in_memory()
            .unwrap();
        Self { dataset }
    }

    pub fn test() -> Self {
        let dataset: InMemDataset<C4Item> = HuggingfaceDatasetLoader::new("c4", "test")
            .extract_string("text")
            .config("en")
            .load_in_memory()
            .unwrap();
        Self { dataset }
    }
}
