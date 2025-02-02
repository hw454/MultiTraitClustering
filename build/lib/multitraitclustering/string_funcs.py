from multitraitclustering import checks as ch

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

def num_to_word(num_orig):
    """Convert integer numbers into their word equivalents.

    Args:
        num_orig (int): Number to be converted

    Returns:
        num_word (string): Space free string describing the number in title case.
    """
    n_dict = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 
              7:'seven',8:'eight', 9:'nine', 10:'ten', 11:'eleven', 12:'twelve',
              13:'thirteen', 14:'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
              18: ' eighteen', 19: 'nineteen', 20: 'twenty'}
    tens_dict = {20: 'twenty', 30: 'thirty', 40:'forty', 50:'fifty', 60:'sixty', 70:'seventy',
                 80: ' eighty', 90: 'ninety', 100:'OneHundred'}
    # Check the input is an integer
    ch.int_check(num_orig, "num_orig")
    # Check whether the number is in a range the function can handle
    if num_orig > 999:
        print("num_to_word does not convert numbers greater than 999 into words.")
        return(str(num_orig))
    # Check the sign of the number before working with the magnitude
    if num_orig < 0:
        num = abs(num_orig)
        sign = "Minus"
    else: 
        num = num_orig
        sign = ""
    # Find the power of ten and the unit separately and combine.
    num_mod_hun = num%100
    if num_mod_hun > 20 and num_mod_hun <= 100:
        digit = num_mod_hun%10
        w1 = n_dict[int(digit)].title()
        ten_mul = num_mod_hun - digit
        w2 = tens_dict[int(ten_mul)].title()
    else:
        w2 = n_dict[int(num_mod_hun)].title()
        w1 = ""
    # If the number is bigger than 100. Find the hundred power and combine with the tens and units.
    if num > 100:
        hundreds = num//100
        w3 = n_dict[int(hundreds)].title()+"Hundred"
    else: w3 = ""
    word_num = sign + w3 + w2 + w1
    return word_num