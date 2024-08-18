import pytest
from ingrain.ingrain_errors import IngrainWebException, error_factory


def test_ingrain_web_exception():
    text = "Sample error"
    status_code = 404
    body = {"error": "Not Found"}

    exception = IngrainWebException(text, status_code, body)

    assert exception.text == text
    assert exception.status_code == status_code
    assert exception.body == body
    assert exception.message == (
        f"Error: {text}. \nStatus Code: {status_code}. \nOriginal Response Body: {body}"
    )
    assert str(exception) == exception.message


@pytest.mark.parametrize(
    "status_code, body",
    [
        (400, {"message": "Bad Request"}),
        (500, {"error": "Internal Server Error"}),
        (403, {"detail": "Forbidden"}),
        (401, {}),
    ],
)
def test_error_factory(status_code, body):
    exception = error_factory(status_code, body)

    assert isinstance(exception, IngrainWebException)
    assert exception.status_code == status_code
    assert exception.body == body
    assert f"Status Code: {status_code}" in exception.message
    assert f"Original Response Body: {body}" in exception.message
    if exception.text:
        assert f"Error: {exception.text}" in exception.message
