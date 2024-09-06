from auto_note import add

def test_given_two_integers_when_add_then_return_sum():
    # Given
    a = 1
    b = 2

    # When
    result = add(a, b)

    # Then
    assert result == 3
