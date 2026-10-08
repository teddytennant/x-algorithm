use crate::processor::record::AuthorId;

pub const FEATURES_VERSION: i64 = 2;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Language {
    len: u8,
    bytes: [u8; Language::CAPACITY],
}

impl Language {
    const CAPACITY: usize = 15;

    pub fn new(code: &str) -> Self {
        let mut language = Self::default();
        if code.len() <= Self::CAPACITY {
            language.bytes[..code.len()].copy_from_slice(code.as_bytes());
            language.len = code.len() as u8;
        }
        language
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PostFeatures {
    pub author_id: AuthorId,
    pub fav_count: i64,
    pub reply_count: i64,
    pub repost_count: i64,
    pub quote_count: i64,
    pub bookmark_count: i64,
    pub view_count: i64,
    pub author_followers_count: i64,
    pub features_ts: i64,
    pub features_version: i64,
    pub has_image: bool,
    pub has_video: bool,
    pub has_media: bool,
    pub is_reply: bool,
    pub is_quote: bool,
    pub author_nsfw_user: bool,
    pub author_nsfw_admin: bool,
    pub language: Language,
}

impl PostFeatures {
    pub fn stamped(now_secs: i64) -> Self {
        Self {
            features_ts: now_secs,
            features_version: FEATURES_VERSION,
            ..Self::default()
        }
    }

    pub fn is_current(&self) -> bool {
        self.features_version >= FEATURES_VERSION
    }
}
