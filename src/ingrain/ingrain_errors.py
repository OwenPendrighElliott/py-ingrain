class IngrainWebException(Exception):
    def __init__(self, text: str, status_code: int, body: dict):
        self.text = text
        self.status_code = status_code
        self.body = body
        self.message = ""
        if self.text is not None:
            self.message += f"Error: {self.text}. \n"
        self.message += (
            f"Status Code: {self.status_code}. \nOriginal Response Body: {self.body}"
        )
        super().__init__(self.message)


def error_factory(status_code: int, body: dict) -> IngrainWebException:
    message = body.get("message")
    if message is None:
        message = body.get("error")
    elif message is None:
        message = body.get("detail")

    return IngrainWebException(message, status_code, body)
