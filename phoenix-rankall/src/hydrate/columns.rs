use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, Int64Array, Int64Builder, StringArray,
    StringBuilder,
};
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;

use super::features::{Language, PostFeatures};

struct Column<T> {
    name: &'static str,
    get: fn(&PostFeatures) -> T,
    set: fn(&mut PostFeatures, T),
}

const INT_COLUMNS: [Column<i64>; 9] = [
    Column {
        name: "fav_count",
        get: |f| f.fav_count,
        set: |f, v| f.fav_count = v,
    },
    Column {
        name: "reply_count",
        get: |f| f.reply_count,
        set: |f, v| f.reply_count = v,
    },
    Column {
        name: "repost_count",
        get: |f| f.repost_count,
        set: |f, v| f.repost_count = v,
    },
    Column {
        name: "quote_count",
        get: |f| f.quote_count,
        set: |f, v| f.quote_count = v,
    },
    Column {
        name: "bookmark_count",
        get: |f| f.bookmark_count,
        set: |f, v| f.bookmark_count = v,
    },
    Column {
        name: "view_count",
        get: |f| f.view_count,
        set: |f, v| f.view_count = v,
    },
    Column {
        name: "author_followers_count",
        get: |f| f.author_followers_count,
        set: |f, v| f.author_followers_count = v,
    },
    Column {
        name: "features_ts",
        get: |f| f.features_ts,
        set: |f, v| f.features_ts = v,
    },
    Column {
        name: "features_version",
        get: |f| f.features_version,
        set: |f, v| f.features_version = v,
    },
];

const BOOL_COLUMNS: [Column<bool>; 7] = [
    Column {
        name: "has_image",
        get: |f| f.has_image,
        set: |f, v| f.has_image = v,
    },
    Column {
        name: "has_video",
        get: |f| f.has_video,
        set: |f, v| f.has_video = v,
    },
    Column {
        name: "has_media",
        get: |f| f.has_media,
        set: |f, v| f.has_media = v,
    },
    Column {
        name: "is_reply",
        get: |f| f.is_reply,
        set: |f, v| f.is_reply = v,
    },
    Column {
        name: "is_quote",
        get: |f| f.is_quote,
        set: |f, v| f.is_quote = v,
    },
    Column {
        name: "author_nsfw_user",
        get: |f| f.author_nsfw_user,
        set: |f, v| f.author_nsfw_user = v,
    },
    Column {
        name: "author_nsfw_admin",
        get: |f| f.author_nsfw_admin,
        set: |f, v| f.author_nsfw_admin = v,
    },
];

const LANGUAGE_COLUMN: &str = "language";

pub fn feature_fields() -> Vec<Field> {
    let ints = INT_COLUMNS
        .iter()
        .map(|c| Field::new(c.name, DataType::Int64, true));
    let bools = BOOL_COLUMNS
        .iter()
        .map(|c| Field::new(c.name, DataType::Boolean, true));
    ints.chain(bools)
        .chain([Field::new(LANGUAGE_COLUMN, DataType::Utf8, true)])
        .collect()
}

pub struct FeatureColumnsBuilder {
    ints: Vec<Int64Builder>,
    bools: Vec<BooleanBuilder>,
    language: StringBuilder,
}

impl FeatureColumnsBuilder {
    pub fn with_capacity(rows: usize) -> Self {
        Self {
            ints: INT_COLUMNS
                .iter()
                .map(|_| Int64Builder::with_capacity(rows))
                .collect(),
            bools: BOOL_COLUMNS
                .iter()
                .map(|_| BooleanBuilder::with_capacity(rows))
                .collect(),
            language: StringBuilder::with_capacity(rows, rows * 2),
        }
    }

    pub fn append(&mut self, features: &PostFeatures) {
        for (builder, column) in self.ints.iter_mut().zip(&INT_COLUMNS) {
            builder.append_value((column.get)(features));
        }
        for (builder, column) in self.bools.iter_mut().zip(&BOOL_COLUMNS) {
            builder.append_value((column.get)(features));
        }
        self.language.append_value(features.language.as_str());
    }

    pub fn finish(mut self) -> Vec<ArrayRef> {
        let ints = self
            .ints
            .iter_mut()
            .map(|b| Arc::new(b.finish()) as ArrayRef);
        let bools = self
            .bools
            .iter_mut()
            .map(|b| Arc::new(b.finish()) as ArrayRef);
        ints.chain(bools)
            .chain([Arc::new(self.language.finish()) as ArrayRef])
            .collect()
    }
}

pub struct FeatureColumnsReader<'a> {
    ints: Vec<Option<&'a Int64Array>>,
    bools: Vec<Option<&'a BooleanArray>>,
    language: Option<&'a StringArray>,
}

impl<'a> FeatureColumnsReader<'a> {
    pub fn new(batch: &'a RecordBatch) -> Self {
        Self {
            ints: INT_COLUMNS.iter().map(|c| column(batch, c.name)).collect(),
            bools: BOOL_COLUMNS.iter().map(|c| column(batch, c.name)).collect(),
            language: column(batch, LANGUAGE_COLUMN),
        }
    }

    pub fn read(&self, row: usize) -> PostFeatures {
        let mut features = PostFeatures::default();
        for (array, column) in self.ints.iter().zip(&INT_COLUMNS) {
            if let Some(array) = array.filter(|a| a.is_valid(row)) {
                (column.set)(&mut features, array.value(row));
            }
        }
        for (array, column) in self.bools.iter().zip(&BOOL_COLUMNS) {
            if let Some(array) = array.filter(|a| a.is_valid(row)) {
                (column.set)(&mut features, array.value(row));
            }
        }
        if let Some(array) = self.language.filter(|a| a.is_valid(row)) {
            features.language = Language::new(array.value(row));
        }
        features
    }
}

pub fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Option<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
}
