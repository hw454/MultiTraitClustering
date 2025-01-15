def format_strings(test_str):
    """Replace digits with words and remove punctuation and spaces

    Args:
        test_str (string): string to be formatted

    Returns:
        test_str (string): formatted string
    """
    test_str = test_str.replace("_","")
    for ele in test_str:
        if ele.isdigit():
            test_str = test_str.replace(ele,  num_to_word(ele).title())
    return test_str

def num_to_word(num):
    """Convert integer numbers into their word equivalents.

    Args:
        num (int): Number to be converted

    Returns:
        num_word (string): Space free string describing the number in title case.
    """
    n_dict = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 
              7:'seven',8:'eight', 9:'nine', 10:'ten', 11:'eleven', 12:'twelve',
              13:'thirteen', 14:'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
              18: ' eighteen', 19: 'nineteen', 20: 'twenty'}
    tens_dict = {20: 'twenty', 30: 'thirty', 40:'forty', 50:'fifty', 60:'sixty', 70:'seventy',
                 80: ' eighty', 90: 'ninety', 100:'OneHundred'}
    if num > 20:
        digit = num%10
        w1 = n_dict[int(digit)].title()
        ten_mul = num - digit
        w2 = tens_dict[int(ten_mul)].title()
        word_num = w1+w2
    else: word_num = n_dict[int(num)].title()
    return word_num