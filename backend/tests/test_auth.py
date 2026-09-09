from app.core.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hashing():
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_jwt_token_generation_and_decoding():
    data = {"sub": "user-uuid-1234", "email": "test@kiavi.ai"}
    token = create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 20
    
    decoded = decode_access_token(token)
    assert decoded.get("sub") == "user-uuid-1234"
    assert decoded.get("email") == "test@kiavi.ai"
    assert "exp" in decoded

if __name__ == "__main__":
    test_password_hashing()
    test_jwt_token_generation_and_decoding()
    print("✅ test_auth passed!")
