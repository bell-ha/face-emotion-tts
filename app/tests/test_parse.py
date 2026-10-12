def test_basic_tags(appmod):
    segs = appmod.parse_tagged_script("안녕하세요.<행복> 비가 오네요.<슬픔> 정말요?")
    assert segs == [("안녕하세요.", "행복"), ("비가 오네요.", "슬픔"), ("정말요?", None)]


def test_spaces_inside_tag_and_newlines(appmod):
    segs = appmod.parse_tagged_script("문장 하나!< 분노 >\n문장 둘?<놀람>\n")
    assert segs == [("문장 하나!", "분노"), ("문장 둘?", "놀람")]


def test_no_tags(appmod):
    assert appmod.parse_tagged_script("  태그 없는 문장.  ") == [("태그 없는 문장.", None)]


def test_empty(appmod):
    assert appmod.parse_tagged_script("") == []
    assert appmod.parse_tagged_script(None) == []


def test_unknown_tag_is_text(appmod):
    # 지원하지 않는 태그(<혐오>)는 본문으로 남는다
    segs = appmod.parse_tagged_script("이상해.<혐오> 좋아.<행복>")
    assert segs == [("이상해.<혐오> 좋아.", "행복")]


def test_consecutive_tags_skip_empty(appmod):
    # 태그 사이에 본문이 없으면 빈 문장은 만들지 않는다(앞 태그 우선)
    segs = appmod.parse_tagged_script("가.<행복><슬픔> 나.<중립>")
    assert segs == [("가.", "행복"), ("나.", "중립")]
