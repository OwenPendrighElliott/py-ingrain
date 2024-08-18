import pytest
from unittest.mock import patch
import pycurl
import json
from io import BytesIO
from ingrain.pycurl_engine import PyCURLEngine


@pytest.fixture
def mock_curl():
    with patch("pycurl.Curl") as MockCurl:
        yield MockCurl.return_value


@pytest.fixture
def engine(mock_curl):
    return PyCURLEngine()


def mock_curl_response(mock_curl, response_data, http_code=200):
    # Create a buffer to simulate the response data
    response_buffer = BytesIO()
    response_buffer.write(json.dumps(response_data).encode("utf-8"))
    response_buffer.seek(0)

    # Mock the getinfo to return the HTTP code
    mock_curl.getinfo.return_value = http_code

    # Capture the WRITEFUNCTION when it's set
    def setopt_side_effect(option, value):
        if option == pycurl.WRITEFUNCTION:
            # Simulate writing the buffer content to the provided write function
            response_content = response_buffer.getvalue()
            for chunk in response_content:
                value(bytes([chunk]))
        return None

    # Mock the perform method to simulate the response
    mock_curl.setopt.side_effect = setopt_side_effect
    mock_curl.perform.side_effect = lambda: None


def test_post(engine, mock_curl):
    response_data = {"success": True}
    mock_curl_response(mock_curl, response_data)

    url = "http://example.com/test"
    data = {"key": "value"}

    response, code = engine.post(url, data)

    assert response == response_data
    assert code == 200
    mock_curl.setopt.assert_any_call(pycurl.URL, url)
    mock_curl.setopt.assert_any_call(pycurl.POSTFIELDS, json.dumps(data))


def test_get(engine, mock_curl):
    response_data = {"data": "value"}
    mock_curl_response(mock_curl, response_data)

    url = "http://example.com/test"

    response, code = engine.get(url)

    assert response == response_data
    assert code == 200
    mock_curl.setopt.assert_any_call(pycurl.URL, url)


def test_put(engine, mock_curl):
    response_data = {"updated": True}
    mock_curl_response(mock_curl, response_data)

    url = "http://example.com/test"
    data = {"update": "value"}

    response, code = engine.put(url, data)

    assert response == response_data
    assert code == 200
    mock_curl.setopt.assert_any_call(pycurl.URL, url)
    mock_curl.setopt.assert_any_call(pycurl.POSTFIELDS, json.dumps(data))


def test_delete(engine, mock_curl):
    response_data = {"deleted": True}
    mock_curl_response(mock_curl, response_data)

    url = "http://example.com/test"

    response, code = engine.delete(url)

    assert response == response_data
    assert code == 200
    mock_curl.setopt.assert_any_call(pycurl.URL, url)


def test_patch(engine, mock_curl):
    response_data = {"patched": True}
    mock_curl_response(mock_curl, response_data)

    url = "http://example.com/test"
    data = {"patch": "value"}

    response, code = engine.patch(url, data)

    assert response == response_data
    assert code == 200
    mock_curl.setopt.assert_any_call(pycurl.URL, url)
    mock_curl.setopt.assert_any_call(pycurl.POSTFIELDS, json.dumps(data))


def test_close(engine, mock_curl):
    engine.close()
    mock_curl.close.assert_called_once()
