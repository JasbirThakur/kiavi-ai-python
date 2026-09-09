from app.services.scraper import clean_domain_name, is_valid_email

def test_clean_domain_name():
    assert clean_domain_name("https://www.alorica.com/about-us") == "alorica.com"
    assert clean_domain_name("http://sub.domain.org/path?query=1") == "sub.domain.org"
    assert clean_domain_name("example.com") == "example.com"

def test_is_valid_email():
    assert is_valid_email("contact@alorica.com") is True
    assert is_valid_email("jasbirsingh17050@gmail.com") is True
    assert is_valid_email("wght@400") is False
    assert is_valid_email("logo@2x.png") is False
    assert is_valid_email("test.com") is False

if __name__ == "__main__":
    test_clean_domain_name()
    test_is_valid_email()
    print("✅ test_scraper passed!")
