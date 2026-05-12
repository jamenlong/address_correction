# Databricks notebook source
# MAGIC %md
# MAGIC ####Malformation Functions

# COMMAND ----------

from pyspark.sql.functions import udf, col, lit, concat, trim
from pyspark.sql.types import *
import random
import numpy as np


# COMMAND ----------

# Malforming addresses

# 1. Replacing characters - DONE
# 2. Repeating characters - DONE
# 3. Writing out numbers - DONE (Consider not using hundreds and thousands)
# 4. Repeating parts of an address - DONE (Consider adding spaces or other separators around numbers when being inserted into numbers Ex: 8530 --> 85853030)
# 5. Adding random words - DONE
# 6. Adding in random text - DONE
# 7. Converting cardinal abreviations to actual words ('W' to 'West' or to 'Wst') - DONE
# 8. Lower-case to upper-case and vice-versa - DONE
# 9. Adding special characters randomly - DONE
# 10. Inserting special characters between numbers (including spaces) - DONE (as per previous task)
# 11. Changing things like 1st to first or 35th to "thirty 5th" - DONE
# 12. Convert number combinations to fractions - Don't remember what this was exactly...
# 13. Randomly add cardinal directions (West, East, etc.) - DONE

# 14. Randomly add apartment numbers - Consider enhancing #5 above to utilize all alphanumeric unit combinations
# 14.b CHANGE EXISTING APT/HOUSE/UNIT NUMBERS TO ANOTHER HOUSE INDICATOR (APT to UNIT OR PARCEL TO TRAILER)

# 15. Randomly insert special characters - DONE (see above)
# 16. Convert abbreviations to full words and vice versa ('Center St' to 'Center Street', 'Center St.' to 'Ctr St.')
# 17. Build function to run through each of these individually on the Smarty dataset. Then build it so that it runs through various combinations of them.
# 18. Find a way to malform Utah addresses. For example, instead of 900 S, write out 9th South or 900 South or 900 Sth or 9th S. - DONE (see above)
# 19. Randomly replace spaces with special characters - DONE

# 20. Randomly remove vowels - DONE
# 21. Capitalize the whole address, and insert lowercase malformations

# COMMAND ----------

# MAGIC %run ./CharacterReplacementDictionary

# COMMAND ----------

# MAGIC %run ./Utilities

# COMMAND ----------

# Character replacement
# NO LONGER NEEDED
# @udf(StringType)
# def replace_char(street_addr, char_malform_dct):
    
#     """
#     The replace_char function will take the street portion of an address, identify the alpha characters in it, and randomly choose up to 50% of those alpha characters to be replaced by a random character from the dictionary of malformed characters. 50% was chosen somewhat arbritrarily. I want to introduce replacement, but not so much so as to fully obscure the address from the delivery driver. Most retail websites have a character limit for inputting addresses, which seems to be around 30-40 characters, so replacing a max of 50% of alpha seems reasonable for a high end.

#     Params:
#         street_addr (str): The street portion of an address. 'Ex: 2124 Sunflower Ave. Apt 2A'
#         char_malorm_dct (dict): A dictionary of malformed characters. Keys are possible alhanumeric characters in the original street addresses. Values are possible malformed versions of those characters. Default for this argument is the char_malform_dct dictionary created above.
#         verbose (bool): If True, print out all output from this function. Default is False.
#     """

#     if verbose: print (f'Original address: {street_addr}')

#     base_alphanum_chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','0']

#     # Get alphanumeric indices of original street address
#     alphanum_indices = [i for i, ltr in enumerate(street_addr) if ltr in base_alphanum_chars]
#     if verbose: print (f'Number of alpha characters in address: {len(alphanum_indices)}')

#     # Choose % of alpha characters in address to be replaced. 
#     # Identify corresponding number of characters
#     max_repl_perc_chars = .5
#     if verbose: print (f'Max % of replacement characters: {max_repl_perc_chars}')

#     # Randomly choose number of replacement characters
#     poss_n_repl_chars = range(int(max_repl_perc_chars * len(alphanum_indices)))
#     cnt_repl_chars = random.choice(poss_n_repl_chars)
#     print (f'Number of replacement characters: {cnt_repl_chars}')

#     # Create new addr which will become malformed
#     malformed_addr = street_addr

#     # Choose random characters to replace with
#     if cnt_repl_chars > 0:

#         for i in range(cnt_repl_chars):

#             # Choose index of random alphanumeric character to replace
#             alphanum_indices = [i for i, ltr in enumerate(malformed_addr) if ltr in base_alphanum_chars]
#             i = list(np.random.choice(alphanum_indices, size=1, replace=False))[0]
#             if verbose: print (f'Randomly chosen index and character to replace: {i}, {malformed_addr[i]}')
            
#             # If alpha char is slected, it needs to be upper to be mapped to possible malformations
#             j = malformed_addr[i].upper() if malformed_addr[i].isalpha() else malformed_addr[i] 

#             malformed_char = random.choice(char_malform_dct[j])
            
#             # malformed_addr = malformed_addr.replace(malformed_addr[i], malformed_char)
#             malformed_addr = malformed_addr[:i] + malformed_char + malformed_addr[i + 1:]

#             if verbose: print (f'Updated malformed addr: {malformed_addr}, original character: {j}, malformed character: {malformed_char}')
#         return malformed_addr
#     else:
#         if verbose: print (f'No alpha characters to replace.')
#         return malformed_addr

    

# COMMAND ----------

# Character replacement

def replace_char(address, char_malform_dct=char_malform_dct, verbose = True):
    """
    The replace_char function will take the street portion of an address, identify the alpha characters in it, and randomly choose up to 50% of those alpha characters to be replaced by a random character from the dictionary of malformed characters. 50% was chosen somewhat arbritrarily. I want to introduce replacement, but not so much so as to fully obscure the address from the delivery driver. Most retail websites have a character limit for inputting addresses, which seems to be around 30-40 characters, so replacing a max of 50% of alpha seems reasonable for a high end.

    Params:
        address (str): The street portion of an address. 'Ex: 2124 Sunflower Ave. Apt 2A'
        char_malform_dct (dict): A dictionary of malformed characters. Keys are possible alhanumeric characters in the original street addresses. Values are possible malformed versions of those characters. Default for this argument is the char_malform_dct dictionary created above.
    """

    # For testing and validation purposes
    output_string = "replace_char"

    if verbose: print (f'Original address: {address}')

    base_alphanum_chars = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','0']

    # Get alphanumeric indices of original street address
    alphanum_indices = [i for i, ltr in enumerate(address) if ltr in base_alphanum_chars]
    if verbose: print (f'Number of alpha characters in address: {len(alphanum_indices)}')

    # Choose % of alpha characters in address to be replaced. 
    # Identify corresponding number of characters
    max_repl_perc_chars = .5
    if verbose: print (f'Max % of replacement characters: {max_repl_perc_chars}')

    # Randomly choose number of replacement characters
    poss_n_repl_chars = int(max_repl_perc_chars * len(alphanum_indices))
    cnt_repl_chars = random.randint(1, poss_n_repl_chars)

    # Create new addr which will become malformed
    malformed_addr = address

    # Choose random characters to replace with
    if cnt_repl_chars > 0:
        for i in range(cnt_repl_chars):
            # Choose index of random alphanumeric character to replace
            alphanum_indices = [i for i, ltr in enumerate(malformed_addr) if ltr in base_alphanum_chars]
            i = list(np.random.choice(alphanum_indices, size=1, replace=False))[0]
            if verbose: print (f'Randomly chosen index and character to replace: {i}, {malformed_addr[i]}')
            
            # If alpha char is selected, it needs to be upper to be mapped to possible malformations
            j = malformed_addr[i].upper() if malformed_addr[i].isalpha() else malformed_addr[i] 

            malformed_char = random.choice(char_malform_dct[j])
            
            malformed_addr = malformed_addr[:i] + malformed_char + malformed_addr[i + 1:]

        if verbose: print (f'Updated malformed addr: {malformed_addr}, original character: {j}, malformed character: {malformed_char}')
        return (malformed_addr, output_string)
    else:
        if verbose: print (f'No alpha characters to replace.')
        return (malformed_addr, output_string)

# Create a UDF for the replace_char function and register it
# replace_char_udf = udf(lambda address: replace_char(address, char_malform_dct, verbose = True), ArrayType(StringType())) 

replace_char_udf = spark.udf.register("replace_char_udf", lambda address: replace_char(address, char_malform_dct, verbose = True), ArrayType(StringType())) 

# COMMAND ----------

def repeat_characters(address, max_chars_repeated = .33333, max_char_repetitions = 7, max_line_length = 50, verbose = True):
    
    """
    repeat_characters will take an address and randomly choose a number of characters to repeat, randomly choose the number of times to repeat those characters, and then repeat those characters accordingly. In order to prevent the address from becoming too long, the number of characters to repeat will be limited to 1/3 the length of the address. Also only allowing a character to be repeated up to 7 times for the same reason.

    Params:
        address: The street portion of a physical address. 'Ex: 2124 Sunflower Ave. Apt 2A'
        max_chars_repeated: Maximum proportion of the original address that will be repeated, as measured in characters. To prevent an address from becoming too long, the default value for this is 1/3. The value entered here will be multiplied by the number of characters in the address.
        max_char_repetitions: Maximum number of times a character can be repeated. To prevent an address from becoming too long, the default value for this is 7.
        max_line_length: Maximum length of an address line. The default value is 50 characters based on the max number of address line characters allowed between USPS, UPS, FedEx and DHL.
        verbose (bool): If True, print out all output from this function. Default is False.
    """    

    import random

    # For testing and validation purposes
    output_string = "repeat_characters"

    # Get lenght of addr
    l = len(address)
    if verbose: print (f'Length of address: {l}')

    # Divide length by 3
    max_n_chars_to_repeat = int(max_chars_repeated * l)
    if verbose: print (f'Max number of characters to repeat: {max_n_chars_to_repeat}')

    # Randomly choose an integer between 1 and 1/3 the length of the address
    n_chars_to_repeat = random.randint(1, max_n_chars_to_repeat)
    if verbose: print (f'Number of characters to repeat: {n_chars_to_repeat}')

    # Copy street address to serve as malformed address that will be iteratively malformed with repeating characters
    malformed_addr = address

    # Choose a character to repeat for each time to repeat (n_chars_to_repeat)
    for i in range(n_chars_to_repeat):
        if verbose: print (f'Loop: {i+1} of {n_chars_to_repeat}')
        
        # Find indices that haven't yet been repeated. This is determined by whether the character is surrounded by the same character or not or leading a string of repeated characters or at the end of a string of repeated characters.
        repeatable_indices = [i for i in range(len(malformed_addr)) \
            if ((i not in [0, 1, len(malformed_addr) - 1, len(malformed_addr) - 2]) \
                and  ((malformed_addr[i] != malformed_addr[i-1]) \
                    and (malformed_addr[i] != malformed_addr[i+1])) \
                and ((malformed_addr[i] != malformed_addr[i-1]) \
                    and (malformed_addr[i] != malformed_addr[i-2]))  \
                and ((malformed_addr[i] != malformed_addr[i+1]) \
                    and (malformed_addr[i] != malformed_addr[i+2]))) \
                
                or ((i == 0) and \
                    (malformed_addr[i] != malformed_addr[i+1]) \
                        and (malformed_addr[i] != malformed_addr[i+2])) \

                or ((i == 1) \
                    and (malformed_addr[i] != malformed_addr[i-1]) \
                        and (malformed_addr[i] != malformed_addr[i+1])) \
                
                or ((i == 1) \
                    and (malformed_addr[i] != malformed_addr[i+1]) \
                        and (malformed_addr[i] != malformed_addr[i+2])) \
                
                or ((i == len(malformed_addr) - 1) \
                    and (malformed_addr[i] != malformed_addr[i-1]) \
                        and (malformed_addr[i] != malformed_addr[i-2])) \
                
                or ((i == len(malformed_addr) - 2) \
                    and (malformed_addr[i] != malformed_addr[i-1]) \
                        and (malformed_addr[i] != malformed_addr[i+1])) \
                
                or ((i == len(malformed_addr) - 2) \
                    and (malformed_addr[i] != malformed_addr[i-1]) \
                        and (malformed_addr[i] != malformed_addr[i-2]))]

        if verbose: print (f'Repeatable indices: {repeatable_indices}')
        
        # Randomly choose an index from indices that are available to be repeated
        i = random.choice(repeatable_indices)
        if verbose: print (f'Randomly chosen index and character to repeat: {i}, {malformed_addr[i]}')

        # Randomly choose how many times the chosen index is to be repeated
        # USPS does not allow address line length to be more than 50 characters, UPS, FedEx, DHL all have fewer character limits. We're going to start with a limit of 50 characters to prove out the functionality.
        total_repeat_char_limit = max_line_length - len(malformed_addr)
        if max_char_repetitions >= total_repeat_char_limit:
            n_times_to_repeat = total_repeat_char_limit
        else:
            n_times_to_repeat = random.randint(1, max_char_repetitions)
        if verbose: print (f'Number of times to repeat: {n_times_to_repeat}')

        # Create repeated string
        repeat_string = malformed_addr[i] * n_times_to_repeat
        if verbose: print (f'Repeat string: {repeat_string}')

        # Replace the character at the randomly-chosen index with a repeated string of the same character
        malformed_addr = malformed_addr[:i] + repeat_string + malformed_addr[i + 1:]
        if verbose: print (f'Updated malformed addr: {malformed_addr}, n_chars: {len(malformed_addr)}\n')

    return (malformed_addr, output_string)

# Create a UDF for the repeat_characters function
# repeat_characters_udf = udf(lambda address: repeat_characters(address, max_chars_repeated = .33333, max_char_repetitions = 7, max_line_length = 50, verbose = True), ArrayType(StringType())) 

repeat_characters_udf = spark.udf.register("repeat_characters_udf", lambda address: repeat_characters(address, max_chars_repeated = .33333, max_char_repetitions = 7, max_line_length = 50, verbose = True), ArrayType(StringType())) 

# COMMAND ----------

# Repeat address parts

def repeat_part_of_address(addr_str, addr_char_lim = 70, max_n_char_insert = 7, verbose = True):

    """
    repeat_part_of_address is a function that will take a string (addr_str), and it will randomly choose a beginning index and an ending index that fall within the length of the string given. It will take the substring that lies between those indices within the original string given (addr_str) and it will insert it back into the addr_string at a randomly-chosen place. The function will return the updated/malformed address string.
    
    Params:
    addr_string (string): address string to be modified
    addr_char_lim (int): maximum number of characters allowed in the output malformed address string
    max_n_char_insert (int): maximum number of characters allowed to be re-inserted into the address string
    verbose (bool): whether to print out intermediate output
    """

    import random

    # For testing and validation purposes
    output_string = "repeat_part_of_address"
    
    addr_char_lim = addr_char_lim
    n_max_char_insert = addr_char_lim - len(addr_str)
    if verbose: print (f'N chars in addr_str: {len(addr_str)}\nN max chars: {addr_char_lim}\nN max chars to insert: {max_n_char_insert}')

    # Randomly choose 2 indices. The smaller is the begin index, the larger is the end index. Must be less than max_n_char_insert. 

    n_char_insert = 999999  # Initially setting this high to force loop to run until it finds a pair of indices than are less than max_n_char_insert. 
    while n_char_insert > max_n_char_insert:
        index1 = random.choice(range(len(addr_str)))
        index2 = random.choice(range(len(addr_str)))

        begin_index = min(index1, index2)
        end_index = max(index1, index2)

        n_char_insert = end_index - begin_index

    if verbose: print (f'Random indices: {index1, index2}\nBegin index: {begin_index}\nEnd index: {end_index}')

    # Get characters between the begin and end indices
    chars_to_insert = addr_str[begin_index:end_index]
    if verbose: print (f'Characters to be inserted: {chars_to_insert}')

    # Randomly choose where to insert the characters that are to be inserted
    insert_index = random.randint(begin_index, end_index)
    if verbose: print (f'Randomly inserted location: {insert_index}')

    # Insert the characters into the address
    malformed_addr = addr_str[:insert_index + 1] + chars_to_insert + addr_str[insert_index:]
    if verbose: print (f"Malformed address: {malformed_addr}")

    # Should strictly-numeric strings be allowed to be inserted into strictly-numeric strings? For example, "8530 Quail Oaks Drive" --> "85853030 Quail Oaks Drive". Maybe consider somehow detecting if numbers are being inserted next to numbers, and if so, trying again until letters are next to numbers. Perhaps best would be to place a space or special character between any pair of numbers generated by inserting one string into another.

    return (malformed_addr, output_string)

# Create a UDF for the repeat_part_of_address function
# repeat_part_of_address_udf = udf(lambda addr_str: repeat_part_of_address(addr_str, addr_char_lim = 70, max_n_char_insert = 7, verbose = True), ArrayType(StringType())) 

repeat_part_of_address_udf = spark.udf.register("repeat_part_of_address_udf", lambda address: repeat_part_of_address(address, addr_char_lim = 70, max_n_char_insert = 7, verbose = True), ArrayType(StringType())) 

# COMMAND ----------


# Insert random words

# 1. Add unit, apt, house, parcel, lot, number, numbers to list
# 1. Get length of address
# 2. Get length limit of address and find max number of characters that can be inserted
# 3. Ientify indices of random words that are less than max number of characters that can be inserted
# 4. Randomly choose one of the random words from the list
# 5. Insert the random word into the address (probably at the end or beginning or where existing spaces exist)
# 6. Consider having spaces or other chracter separators before and after the random word


# COMMAND ----------

# This was put into a separate notebook that is run using %run. See above.
# NO LONGER NEEDED

# holiday_lst = ["birthday", "christmas", "easter", "new years", "valentines", "thanksgiving", "halloween", "mothers day", "fathers day", "memorial day", "independence day", "holiday"]

# event_lst = ["baby", "graduation", "celebration", "marriage", "divorce", "party", "wedding", "birth", "baptism", "baptize", "sacrament", "communion", "bar mitzvah", "bat mitzvah", "hanukkah", "hanukkah mitzvah", "hanukkah bat mitzvah", "hanukkah holiday", "hanukkah holiday mitzvah", "hanukkah holiday bat mitzvah", "hanukkah holiday bar mitzvah", "hanukkah holiday bat mitzvah bar mitzvah", "hanukkah holiday bar mitzvah bat mitzvah"]

# gc_msg_lst = ["congratulations", "happy birthday", "happy easter", "happy halloween", "happy thanksgiving", "happy valentines day", "happy mothers day", "happy fathers day", "happy memorial day", "happy independence day", "happy new years", "happy halloween", "happy thanksgiving", "happy valentines day", "happy mothers day", "happy fathers day", "happy memorial day", "sorry for your loss", "we miss you", "come visit us", "soon", "hope", "miss you", "uncle", "cousin", "aunt", "newphew", "niece", "friend", "companion", "dog", "cat", "pet", "animal", "grand mother", "grand father", "grand son", "grand daughter", "grand baby", "grandma", "grandpa", "friend", "classmate", "colleage", "coworker", "boss", "manager", "director"]

# interior_house_lst = ["living room", "family room", "room", "rm", "guest room", "bedroom", "bathroom", "guest bath", "kitchen", "attic", "closet", "closet", "closets", "closet door", "closet doors", "closet unit", "closet units", "closet apartment", "floor", "suite", "ste", "upstairs", "downstairs", "basement", "light switch", "ceiling fan", "sun room", "atrium", "lounge", "loft", "oven", "microwave", "refrigerator"]

# exterior_house_lst = ["patio", "back patio", "porch", "front patio", "deck", "front stairs", "back stairs", "mailbox", "garage", "garage door", "garage doors", "garage unit", "garage units", "garage apartment", "garage apartments", "garage room", "garage rooms", "guest", "guest house", "guest houses", "guest suite", "guest suites", "guest apartment", "guest apartments", "guest room", "guest rooms", "guesthouse", "guesthouses", "guest suite", "guest suites", "shed", "barn", "workshop", "driveway", "pool", "pool house", "garden", "backyard", "front yard", "side house", "side yard", "yard", "roof", "steps", "build", "carport", "parking", "parking space", "gate", "fence", "border", "property", "marker", "sign", "sign post", "light post", "street light", "grass", "lawn", "flowers", "landscape", "wall"]

# floor_lst = ["ground floor", "first floor", "second floor", "third floor", "fourth floor", "fifth floor", "sixth floor", "seventh floor", "eighth floor", "ninth floor", "tenth floor", "eleventh floor", "twelfth floor", "thirteenth floor", "fourteenth floor", "fifteenth floor", "sixteenth floor", "seventeenth floor", "eighteenth floor", "nineteenth floor", "twentieth floor", "twenty first floor", "twenty second floor", "twenty third floor", "twenty fourth floor", "twenty fifth floor", "twenty sixth floor", "twenty seventh floor", "twenty eighth floor", "twenty ninth floor", "thirtieth floor", "thirty first floor", "thirty second floor", "thirty third floor", "thirty fourth floor", "thirty fifth floor", "thirty sixth floor", "thirty seventh floor", "thirty eighth floor", "thirty ninth floor", "fortieth floor", "forty first floor", "forty second floor", "forty third floor", "forty fourth floor", "forty fifth floor", "forty sixth floor", "forty seventh floor", "forty eighth floor", "forty ninth floor", "fiftieth floor", "fifty first floor", "fifty second floor", "fifty third floor", "fifty fourth floor", "fifty fifth floor", "fifty sixth floor", "fifty seventh floor", "fifty eighth floor", "fifty ninth floor", "sixtieth floor", "sixty first floor", "sixty second floor", "sixty third floor", "sixty fourth floor", "sixty fifth floor", "sixty sixth floor", "sixty seventh floor", "sixty eighth floor", "sixty ninth floor", "seventieth floor", "seventy first floor", "seventy second floor", "seventy third floor", "seventy fourth floor", "seventy fifth floor", "seventy sixth floor", "seventy seventh floor", "seventy eighth floor", "seventy ninth floor", "eightieth floor", "eighty first floor", "eighty second floor", "eighty third floor", "eighty fourth floor", "eighty fifth floor", "eighty sixth floor", "eighty seventh floor", "eighty eighth floor", "eighty ninth floor", "ninetieth floor", "ninety first floor", "ninety second floor", "ninety third floor", "ninety fourth floor", "ninety fifth floor", "ninety sixth floor", "ninety seventh floor", "ninety eighth floor", "ninety ninth floor"]

# unit_lst = ["ground unit", "first unit", "second unit", "third unit", "fourth unit", "fifth unit", "sixth unit", "seventh unit", "eighth unit", "ninth unit", "tenth unit", "eleventh unit", "twelfth unit", "thirteenth unit", "fourteenth unit", "fifteenth unit", "sixteenth unit", "seventeenth unit", "eighteenth unit", "nineteenth unit", "twentieth unit", "twenty first unit", "twenty second unit", "twenty third unit", "twenty fourth unit", "twenty fifth unit", "twenty sixth unit", "twenty seventh unit", "twenty eighth unit", "twenty ninth unit", "thirtieth unit", "thirty first unit", "thirty second unit", "thirty third unit", "thirty fourth unit", "thirty fifth unit", "thirty sixth unit", "thirty seventh unit", "thirty eighth unit", "thirty ninth unit", "fortieth unit", "forty first unit", "forty second unit", "forty third unit", "forty fourth unit", "forty fifth unit", "forty sixth unit", "forty seventh unit", "forty eighth unit", "forty ninth unit", "fiftieth unit", "fifty first unit", "fifty second unit", "fifty third unit", "fifty fourth unit", "fifty fifth unit", "fifty sixth unit", "fifty seventh unit", "fifty eighth unit", "fifty ninth unit", "sixtieth unit", "sixty first unit", "sixty second unit", "sixty third unit", "sixty fourth unit", "sixty fifth unit", "sixty sixth unit", "sixty seventh unit", "sixty eighth unit", "sixty ninth unit", "seventieth unit", "seventy first unit", "seventy second unit", "seventy third unit", "seventy fourth unit", "seventy fifth unit", "seventy sixth unit", "seventy seventh unit", "seventy eighth unit", "seventy ninth unit", "eightieth unit", "eighty first unit", "eighty second unit", "eighty third unit", "eighty fourth unit", "eighty fifth unit", "eighty sixth unit", "eighty seventh unit", "eighty eighth unit", "eighty ninth unit", "ninetieth unit", "ninety first unit", "ninety second unit", "ninety third unit", "ninety fourth unit", "ninety fifth unit", "ninety sixth unit", "ninety seventh unit", "ninety eighth unit", "ninety ninth unit", "unit 1", "unit 2", "unit 3", "unit 4", "unit 5", "unit 6", "unit 7", "unit 8", "unit 9", "unit 10", "unit 11", "unit 12", "unit 13", "unit 14", "unit 15", "unit 16", "unit 17", "unit 18", "unit 19", "unit 20", "unit 21", "unit 22", "unit 23", "unit 24", "unit 25", "unit 26", "unit 27", "unit 28", "unit 29", "unit 30", "unit 31", "unit 32", "unit 33", "unit 34", "unit 35", "unit 36", "unit 37", "unit 38", "unit 39", "unit 40", "unit 41", "unit 42", "unit 43", "unit 44", "unit 45", "unit 46", "unit 47", "unit 48", "unit 49", "unit 50", "unit 51", "unit 52", "unit 53", "unit 54", "unit 55", "unit 56", "unit 57", "unit 58", "unit 59", "unit 60", "unit 61", "unit 62", "unit 63", "unit 64", "unit 65", "unit 66", "unit 67", "unit 68", "unit 69", "unit 70", "unit 71", "unit 72", "unit 73", "unit 74", "unit 75", "unit 76", "unit 77", "unit 78", "unit 79", "unit 80", "unit 81", "unit 82", "unit 83", "unit 84", "unit 85", "unit 86", "unit 87", "unit 88", "unit 89", "unit 90", "unit 91", "unit 92", "unit 93", "unit 94", "unit 95", "unit 96", "unit 97", "unit 98", "unit 99", "unit 100", "unit A", "unit B", "unit C", "unit D", "unit E", "unit F", "unit G", "unit H", "unit I", "unit J", "unit K", "unit L", "unit M", "unit N", "unit O", "unit P", "unit Q", "unit R", "unit S", "unit T", "unit U", "unit V", "unit W", "unit X", "unit Y", "unit Z", "unit AA", "unit AB", "unit AC", "unit AD", "unit AE", "unit AF", "unit AG", "unit AH", "unit AI", "unit AJ", "unit AK", "unit AL", "unit AM", "unit AN", "unit AO", "unit AP", "unit AQ", "unit AR", "unit AS", "unit AT", "unit AU", "unit AV", "unit AW", "unit AX", "unit AY", "unit AZ"]

# apt_lst = ["ground apt", "first apt", "second apt", "third apt", "fourth apt", "fifth apt", "sixth apt", "seventh apt", "eighth apt", "ninth apt", "tenth apt", "eleventh apt", "twelfth apt", "thirteenth apt", "fourteenth apt", "fifteenth apt", "sixteenth apt", "seventeenth apt", "eighteenth apt", "nineteenth apt", "twentieth apt", "twenty first apt", "twenty second apt", "twenty third apt", "twenty fourth apt", "twenty fifth apt", "twenty sixth apt", "twenty seventh apt", "twenty eighth apt", "twenty ninth apt", "thirtieth apt", "thirty first apt", "thirty second apt", "thirty third apt", "thirty fourth apt", "thirty fifth apt", "thirty sixth apt", "thirty seventh apt", "thirty eighth apt", "thirty ninth apt", "fortieth apt", "forty first apt", "forty second apt", "forty third apt", "forty fourth apt", "forty fifth apt", "forty sixth apt", "forty seventh apt", "forty eighth apt", "forty ninth apt", "fiftieth apt", "fifty first apt", "fifty second apt", "fifty third apt", "fifty fourth apt", "fifty fifth apt", "fifty sixth apt", "fifty seventh apt", "fifty eighth apt", "fifty ninth apt", "sixtieth apt", "sixty first apt", "sixty second apt", "sixty third apt", "sixty fourth apt", "sixty fifth apt", "sixty sixth apt", "sixty seventh apt", "sixty eighth apt", "sixty ninth apt", "seventieth apt", "seventy first apt", "seventy second apt", "seventy third apt", "seventy fourth apt", "seventy fifth apt", "seventy sixth apt", "seventy seventh apt", "seventy eighth apt", "seventy ninth apt", "eightieth apt", "eighty first apt", "eighty second apt", "eighty third apt", "eighty fourth apt", "eighty fifth apt", "eighty sixth apt", "eighty seventh apt", "eighty eighth apt", "eighty ninth apt", "ninetieth apt", "ninety first apt", "ninety second apt", "ninety third apt", "ninety fourth apt", "ninety fifth apt", "ninety sixth apt", "ninety seventh apt", "ninety eighth apt", "ninety ninth apt", "apt 1", "apt 2", "apt 3", "apt 4", "apt 5", "apt 6", "apt 7", "apt 8", "apt 9", "apt 10", "apt 11", "apt 12", "apt 13", "apt 14", "apt 15", "apt 16", "apt 17", "apt 18", "apt 19", "apt 20", "apt 21", "apt 22", "apt 23", "apt 24", "apt 25", "apt 26", "apt 27", "apt 28", "apt 29", "apt 30", "apt 31", "apt 32", "apt 33", "apt 34", "apt 35", "apt 36", "apt 37", "apt 38", "apt 39", "apt 40", "apt 41", "apt 42", "apt 43", "apt 44", "apt 45", "apt 46", "apt 47", "apt 48", "apt 49", "apt 50", "apt 51", "apt 52", "apt 53", "apt 54", "apt 55", "apt 56", "apt 57", "apt 58", "apt 59", "apt 60", "apt 61", "apt 62", "apt 63", "apt 64", "apt 65", "apt 66", "apt 67", "apt 68", "apt 69", "apt 70", "apt 71", "apt 72", "apt 73", "apt 74", "apt 75", "apt 76", "apt 77", "apt 78", "apt 79", "apt 80", "apt 81", "apt 82", "apt 83", "apt 84", "apt 85", "apt 86", "apt 87", "apt 88", "apt 89", "apt 90", "apt 91", "apt 92", "apt 93", "apt 94", "apt 95", "apt 96", "apt 97", "apt 98", "apt 99", "apt 100", "apt A", "apt B", "apt C", "apt D", "apt E", "apt F", "apt G", "apt H", "apt I", "apt J", "apt K", "apt L", "apt M", "apt N", "apt O", "apt P", "apt Q", "apt R", "apt S", "apt T", "apt U", "apt V", "apt W", "apt X", "apt Y", "apt Z", "apt AA", "apt AB", "apt AC", "apt AD", "apt AE", "apt AF", "apt AG", "apt AH", "apt AI", "apt AJ", "apt AK", "apt AL", "apt AM", "apt AN", "apt AO", "apt AP", "apt AQ", "apt AR", "apt AS", "apt AT", "apt AU", "apt AV", "apt AW", "apt AX", "apt AY", "apt AZ"]

# house_lst = ["house 1", "house 2", "house 3", "house 4", "house 5", "house 6", "house 7", "house 8", "house 9", "house 10", "house 11", "house 12", "house 13", "house 14", "house 15", "house 16", "house 17", "house 18", "house 19", "house 20", "house 21", "house 22", "house 23", "house 24", "house 25", "house 26", "house 27", "house 28", "house 29", "house 30", "house 31", "house 32", "house 33", "house 34", "house 35", "house 36", "house 37", "house 38", "house 39", "house 40", "house 41", "house 42", "house 43", "house 44", "house 45", "house 46", "house 47", "house 48", "house 49", "house 50", "house 51", "house 52", "house 53", "house 54", "house 55", "house 56", "house 57", "house 58", "house 59", "house 60", "house 61", "house 62", "house 63", "house 64", "house 65", "house 66", "house 67", "house 68", "house 69", "house 70", "house 71", "house 72", "house 73", "house 74", "house 75", "house 76", "house 77", "house 78", "house 79", "house 80", "house 81", "house 82", "house 83", "house 84", "house 85", "house 86", "house 87", "house 88", "house 89", "house 90", "house 91", "house 92", "house 93", "house 94", "house 95", "house 96", "house 97", "house 98", "house 99", "house 100", "house A", "house B", "house C", "house D", "house E", "house F", "house G", "house H", "house I", "house J", "house K", "house L", "house M", "house N", "house O", "house P", "house Q", "house R", "house S", "house T", "house U", "house V", "house W", "house X", "house Y", "house Z", "house AA", "house AB", "house AC", "house AD", "house AE", "house AF", "house AG", "house AH", "house AI", "house AJ", "house AK", "house AL", "house AM", "house AN", "house AO", "house AP", "house AQ", "house AR", "house AS", "house AT", "house AU", "house AV", "house AW", "house AX", "house AY", "house AZ"]

# lot_lst = ["lot 1", "lot 2", "lot 3", "lot 4", "lot 5", "lot 6", "lot 7", "lot 8", "lot 9", "lot 10", "lot 11", "lot 12", "lot 13", "lot 14", "lot 15", "lot 16", "lot 17", "lot 18", "lot 19", "lot 20", "lot 21", "lot 22", "lot 23", "lot 24", "lot 25", "lot 26", "lot 27", "lot 28", "lot 29", "lot 30", "lot 31", "lot 32", "lot 33", "lot 34", "lot 35", "lot 36", "lot 37", "lot 38", "lot 39", "lot 40", "lot 41", "lot 42", "lot 43", "lot 44", "lot 45", "lot 46", "lot 47", "lot 48", "lot 49", "lot 50", "lot 51", "lot 52", "lot 53", "lot 54", "lot 55", "lot 56", "lot 57", "lot 58", "lot 59", "lot 60", "lot 61", "lot 62", "lot 63", "lot 64", "lot 65", "lot 66", "lot 67", "lot 68", "lot 69", "lot 70", "lot 71", "lot 72", "lot 73", "lot 74", "lot 75", "lot 76", "lot 77", "lot 78", "lot 79", "lot 80", "lot 81", "lot 82", "lot 83", "lot 84", "lot 85", "lot 86", "lot 87", "lot 88", "lot 89", "lot 90", "lot 91", "lot 92", "lot 93", "lot 94", "lot 95", "lot 96", "lot 97", "lot 98", "lot 99", "lot 100", "lot A", "lot B", "lot C", "lot D", "lot E", "lot F", "lot G", "lot H", "lot I", "lot J", "lot K", "lot L", "lot M", "lot N", "lot O", "lot P", "lot Q", "lot R", "lot S", "lot T", "lot U", "lot V", "lot W", "lot X", "lot Y", "lot Z", "lot AA", "lot AB", "lot AC", "lot AD", "lot AE", "lot AF", "lot AG", "lot AH", "lot AI", "lot AJ", "lot AK", "lot AL", "lot AM", "lot AN", "lot AO", "lot AP", "lot AQ", "lot AR", "lot AS", "lot AT", "lot AU", "lot AV", "lot AW", "lot AX", "lot AY", "lot AZ"]

# number_lst = ["number 1", "number 2", "number 3", "number 4", "number 5", "number 6", "number 7", "number 8", "number 9", "number 10", "number 11", "number 12", "number 13", "number 14", "number 15", "number 16", "number 17", "number 18", "number 19", "number 20", "number 21", "number 22", "number 23", "number 24", "number 25", "number 26", "number 27", "number 28", "number 29", "number 30", "number 31", "number 32", "number 33", "number 34", "number 35", "number 36", "number 37", "number 38", "number 39", "number 40", "number 41", "number 42", "number 43", "number 44", "number 45", "number 46", "number 47", "number 48", "number 49", "number 50", "number 51", "number 52", "number 53", "number 54", "number 55", "number 56", "number 57", "number 58", "number 59", "number 60", "number 61", "number 62", "number 63", "number 64", "number 65", "number 66", "number 67", "number 68", "number 69", "number 70", "number 71", "number 72", "number 73", "number 74", "number 75", "number 76", "number 77", "number 78", "number 79", "number 80", "number 81", "number 82", "number 83", "number 84", "number 85", "number 86", "number 87", "number 88", "number 89", "number 90", "number 91", "number 92", "number 93", "number 94", "number 95", "number 96", "number 97", "number 98", "number 99", "number 100"]

# trailer_lst = ["trailer 1", "trailer 2", "trailer 3", "trailer 4", "trailer 5", "trailer 6", "trailer 7", "trailer 8", "trailer 9", "trailer 10", "trailer 11", "trailer 12", "trailer 13", "trailer 14", "trailer 15", "trailer 16", "trailer 17", "trailer 18", "trailer 19", "trailer 20", "trailer 21", "trailer 22", "trailer 23", "trailer 24", "trailer 25", "trailer 26", "trailer 27", "trailer 28", "trailer 29", "trailer 30", "trailer 31", "trailer 32", "trailer 33", "trailer 34", "trailer 35", "trailer 36", "trailer 37", "trailer 38", "trailer 39", "trailer 40", "trailer 41", "trailer 42", "trailer 43", "trailer 44", "trailer 45", "trailer 46", "trailer 47", "trailer 48", "trailer 49", "trailer 50", "trailer 51", "trailer 52", "trailer 53", "trailer 54", "trailer 55", "trailer 56", "trailer 57", "trailer 58", "trailer 59", "trailer 60", "trailer 61", "trailer 62", "trailer 63", "trailer 64", "trailer 65", "trailer 66", "trailer 67", "trailer 68", "trailer 69", "trailer 70", "trailer 71", "trailer 72", "trailer 73", "trailer 74", "trailer 75", "trailer 76", "trailer 77", "trailer 78", "trailer 79", "trailer 80", "trailer 81", "trailer 82", "trailer 83", "trailer 84", "trailer 85", "trailer 86", "trailer 87", "trailer 88", "trailer 89", "trailer 90", "trailer 91", "trailer 92", "trailer 93", "trailer 94", "trailer 95", "trailer 96", "trailer 97", "trailer 98", "trailer 99", "trailer 100", "trailer A", "trailer B", "trailer C", "trailer D", "trailer E", "trailer F", "trailer G", "trailer H", "trailer I", "trailer J", "trailer K", "trailer L", "trailer M", "trailer N", "trailer O", "trailer P", "trailer Q", "trailer R", "trailer S", "trailer T", "trailer U", "trailer V", "trailer W", "trailer X", "trailer Y", "trailer Z", "trailer AA", "trailer AB", "trailer AC", "trailer AD", "trailer AE", "trailer AF", "trailer AG", "trailer AH", "trailer AI", "trailer AJ", "trailer AK", "trailer AL", "trailer AM", "trailer AN", "trailer AO", "trailer AP", "trailer AQ", "trailer AR", "trailer AS", "trailer AT", "trailer AU", "trailer AV", "trailer AW", "trailer AX", "trailer AY", "trailer AZ"]

# plot_lst = ["plot 1", "plot 2", "plot 3", "plot 4", "plot 5", "plot 6", "plot 7", "plot 8", "plot 9", "plot 10", "plot 11", "plot 12", "plot 13", "plot 14", "plot 15", "plot 16", "plot 17", "plot 18", "plot 19", "plot 20", "plot 21", "plot 22", "plot 23", "plot 24", "plot 25", "plot 26", "plot 27", "plot 28", "plot 29", "plot 30", "plot 31", "plot 32", "plot 33", "plot 34", "plot 35", "plot 36", "plot 37", "plot 38", "plot 39", "plot 40", "plot 41", "plot 42", "plot 43", "plot 44", "plot 45", "plot 46", "plot 47", "plot 48", "plot 49", "plot 50", "plot 51", "plot 52", "plot 53", "plot 54", "plot 55", "plot 56", "plot 57", "plot 58", "plot 59", "plot 60", "plot 61", "plot 62", "plot 63", "plot 64", "plot 65", "plot 66", "plot 67", "plot 68", "plot 69", "plot 70", "plot 71", "plot 72", "plot 73", "plot 74", "plot 75", "plot 76", "plot 77", "plot 78", "plot 79", "plot 80", "plot 81", "plot 82", "plot 83", "plot 84", "plot 85", "plot 86", "plot 87", "plot 88", "plot 89", "plot 90", "plot 91", "plot 92", "plot 93", "plot 94", "plot 95", "plot 96", "plot 97", "plot 98", "plot 99", "plot 100", "plot A", "plot B", "plot C", "plot D", "plot E", "plot F", "plot G", "plot H", "plot I", "plot J", "plot K", "plot L", "plot M", "plot N", "plot O", "plot P", "plot Q", "plot R", "plot S", "plot T", "plot U", "plot V", "plot W", "plot X", "plot Y", "plot Z", "plot AA", "plot AB", "plot AC", "plot AD", "plot AE", "plot AF", "plot AG", "plot AH", "plot AI", "plot AJ", "plot AK", "plot AL", "plot AM", "plot AN", "plot AO", "plot AP", "plot AQ", "plot AR", "plot AS", "plot AT", "plot AU", "plot AV", "plot AW", "plot AX", "plot AY", "plot AZ"]

# parcel_lst = ["parcel 1", "parcel 2", "parcel 3", "parcel 4", "parcel 5", "parcel 6", "parcel 7", "parcel 8", "parcel 9", "parcel 10", "parcel 11", "parcel 12", "parcel 13", "parcel 14", "parcel 15", "parcel 16", "parcel 17", "parcel 18", "parcel 19", "parcel 20", "parcel 21", "parcel 22", "parcel 23", "parcel 24", "parcel 25", "parcel 26", "parcel 27", "parcel 28", "parcel 29", "parcel 30", "parcel 31", "parcel 32", "parcel 33", "parcel 34", "parcel 35", "parcel 36", "parcel 37", "parcel 38", "parcel 39", "parcel 40", "parcel 41", "parcel 42", "parcel 43", "parcel 44", "parcel 45", "parcel 46", "parcel 47", "parcel 48", "parcel 49", "parcel 50", "parcel 51", "parcel 52", "parcel 53", "parcel 54", "parcel 55", "parcel 56", "parcel 57", "parcel 58", "parcel 59", "parcel 60", "parcel 61", "parcel 62", "parcel 63", "parcel 64", "parcel 65", "parcel 66", "parcel 67", "parcel 68", "parcel 69", "parcel 70", "parcel 71", "parcel 72", "parcel 73", "parcel 74", "parcel 75", "parcel 76", "parcel 77", "parcel 78", "parcel 79", "parcel 80", "parcel 81", "parcel 82", "parcel 83", "parcel 84", "parcel 85", "parcel 86", "parcel 87", "parcel 88", "parcel 89", "parcel 90", "parcel 91", "parcel 92", "parcel 93", "parcel 94", "parcel 95", "parcel 96", "parcel 97", "parcel 98", "parcel 99", "parcel 100", "parcel A", "parcel B", "parcel C", "parcel D", "parcel E", "parcel F", "parcel G", "parcel H", "parcel I", "parcel J", "parcel K", "parcel L", "parcel M", "parcel N", "parcel O", "parcel P", "parcel Q", "parcel R", "parcel S", "parcel T", "parcel U", "parcel V", "parcel W", "parcel X", "parcel Y", "parcel Z", "parcel AA", "parcel AB", "parcel AC", "parcel AD", "parcel AE", "parcel AF", "parcel AG", "parcel AH", "parcel AI", "parcel AJ", "parcel AK", "parcel AL", "parcel AM", "parcel AN", "parcel AO", "parcel AP", "parcel AQ", "parcel AR", "parcel AS", "parcel AT", "parcel AU", "parcel AV", "parcel AW", "parcel AX", "parcel AY", "parcel AZ"]

# room_lst = ["room 1", "room 2", "room 3", "room 4", "room 5", "room 6", "room 7", "room 8", "room 9", "room 10", "room 11", "room 12", "room 13", "room 14", "room 15", "room 16", "room 17", "room 18", "room 19", "room 20", "room 21", "room 22", "room 23", "room 24", "room 25", "room 26", "room 27", "room 28", "room 29", "room 30", "room 31", "room 32", "room 33", "room 34", "room 35", "room 36", "room 37", "room 38", "room 39", "room 40", "room 41", "room 42", "room 43", "room 44", "room 45", "room 46", "room 47", "room 48", "room 49", "room 50", "room 51", "room 52", "room 53", "room 54", "room 55", "room 56", "room 57", "room 58", "room 59", "room 60", "room 61", "room 62", "room 63", "room 64", "room 65", "room 66", "room 67", "room 68", "room 69", "room 70", "room 71", "room 72", "room 73", "room 74", "room 75", "room 76", "room 77", "room 78", "room 79", "room 80", "room 81", "room 82", "room 83", "room 84", "room 85", "room 86", "room 87", "room 88", "room 89", "room 90", "room 91", "room 92", "room 93", "room 94", "room 95", "room 96", "room 97", "room 98", "room 99", "room 100", "room A", "room B", "room C", "room D", "room E", "room F", "room G", "room H", "room I", "room J", "room K", "room L", "room M", "room N", "room O", "room P", "room Q", "room R", "room S", "room T", "room U", "room V", "room W", "room X", "room Y", "room Z", "room AA", "room AB", "room AC", "room AD", "room AE", "room AF", "room AG", "room AH", "room AI", "room AJ", "room AK", "room AL", "room AM", "room AN", "room AO", "room AP", "room AQ", "room AR", "room AS", "room AT", "room AU", "room AV", "room AW", "room AX", "room AY", "room AZ"]

# street_type_lst = ["street", "strt", "drive", "dr", "avenue", "ave", "road", "rd", "boulevard", "blvd", "highway", "hwy", "lane", "ln", "way", "wy", "terrace", "ter", "place", "pl", "square", "sq", "court", "ct", "circle", "cir", "parkway", "pkwy", "ter", "cul de sac"]

# random_word_lst = ["college", "university", "saw mill", "pavement", "cement", "concrete", "sidewalk", "owner", "mortgage", "location", "red", "orange", "yellow", "green", "blue", "purple", "violet", "silver", "gold", "black", "white", "brown", "gray", "tan", "charcoal", "grill", "sandbox", "playground", "window", "postmaster", "mailman", "driver", "delivery", "box", "package", "purchase", "truck", "car", "van", "sedan", "coupe", "honda", "toyota", "ford", "chevy", "dodge", "ram", "jeep", "chrysler", "nissan", "mazda", "subaru", "mercedes", "bmw", "audi", "lexus", "volkswagen", "infiniti", "kia", "boat", "shoes", "clothes", "jeans", "pants", "shirts", "hamburger", "money", "payment", "credit", "kids", "roommate", "return", "warehouse", "whse", "individual", "UPS", "FedEx", "DHL", "USPS", "Amazon", "eBay", "GOAT", "StockX", "Nike", "Retail", "Expedite", "Standard", "Ship", "Shipping", "Service", "computer", "address", "warranty", "return", "stock", "athlete", "sports", "buy", "receipt", "packing list"]

# random_word_lst = holiday_lst + event_lst + gc_msg_lst + interior_house_lst + exterior_house_lst + floor_lst + unit_lst + apt_lst + house_lst + lot_lst + number_lst + trailer_lst + plot_lst + parcel_lst + room_lst + street_type_lst + random_word_lst

# COMMAND ----------

# Insert random words

# 1. Add unit, apt, house, parcel, lot, number, numbers to list
# 1. Get length of address
# 2. Get length limit of address and find max number of characters that can be inserted
# 3. Ientify indices of random words that are less than max number of characters that can be inserted
# 4. Randomly choose one of the random words from the list
# 5. Insert the random word into the address (probably at the end or beginning or where existing spaces exist)
# 6. Consider having spaces or other chracter separators before and after the random word

def insert_random_words(address, random_word_list, address_char_lim = 70, verbose = True):
    
    """
    insert_random_words is a function that will insert a random word or words into an address string. It will consider how many characters are already in an address string, as well as the maximum number of characters that can exist in an address string and will fill the address string with a random word or words such that the end result string has the maximum allowed characters or less.

    Params:
        address (str): The address string to be modified.
        random_word_list (lst): List of random words to serve as optional strings to be inserted into the address string.
        address_char_lim (int): Maximum number of characters allowed in the address string. Default is 70.
        verbose (bool): Boolean flag to print out statements showing function progress. Default is True.
    """

    # For testing and validating purposes
    output_string = "insert_random_words"
    
    # Get length of address string given
    address_length = len(address)
    if verbose: print (f"Inserting random words...")
    if verbose: print (f"Length of input address: {address_length}")

    # Calculate the max number of characters than can be inserted into the address string and still be less than or equal to the max number of characters allowed in the address string. Also subtracting 1 to account for a random space/separator character that will be inserted.
    max_insertion_string_length = address_char_lim - address_length - 1
    if verbose: print (f"Max insertion string length: {max_insertion_string_length}")

    # Given the max insertion string length, identify possible random strings that can be inserted into the address string and still be less than the address character limit. Also includes criteria such that if a random word already exists in the address string, it will not be chosen as a random word to be inserted into the address string. The reason for this is that we don't want an address that looks like 1234 Main St. Apt A Apt D.
    random_word_options = [w for w in random_word_list if (len(w) <= max_insertion_string_length) & (w[0].lower() not in address.lower())] + [""]
    if verbose: print (f"Number of random word options: {len(random_word_options)}")
    if verbose: print (f"Sample of random word options: {random_word_options[:5]}")

    # Randomly choose one of the random strings that can be inserted into the address string.
    random_insertion_word = random.choice(random_word_options)
    if verbose: print (f"Randomly chosen insertion word: {random_insertion_word}")

    # Instantiates a list of separator options
    # Chooses one at random. If the address is already at the max character length allowed, no separator is inserted.
    separator_options = [" ", "-", "_", ":", ".", ",", "/", "!", "?", "(", ")", "[", "]", "{", "}", "|", "@"]
    separator = "" if random_insertion_word == "" else random.choice(separator_options)
    if verbose: print (f"Randomly chosen separator: {separator}")

    # Insert the random word into the address string.
    output_address = address + separator + random_insertion_word
    if verbose: print (f"Malformed address: {output_address}")

    return (output_address, output_string)

# Create a UDF for the insert_random_words function
# insert_random_words_udf = udf(lambda address: insert_random_words(address, random_word_list = random_word_lst, address_char_lim = 70, verbose = True), ArrayType(StringType())) 

insert_random_words_udf = spark.udf.register("insert_random_words_udf", lambda address: insert_random_words(address, random_word_list = random_word_lst, address_char_lim = 70, verbose = True), ArrayType(StringType())) 

# COMMAND ----------

# Add random text  

def add_random_text(address, address_char_lim = 70, verbose = True):

    """
    add_random_text will take an input address string and add random text. The text chosen is entirely random. First, the length of the address is calculated, then the difference between the maximum character limit allowed in an address string and the address is provided. Then, a random number between 0 and that difference is chosen which will serve as the length of the random text. After that, random characters are chosen one-by-one until the intended length of the random text is achieved. Once the random text is chosen, it is inserted into the address string randomly. The final address with random text is returned.

    Params:
        address (str): The address string to be modified.
        address_char_lim (int): Maximum number of characters allowed in the address string. Default is 70.
        verbose (bool): Boolean flag used to print statements showing function progress . Default is True.
    """

    import random
    import string

    # For testing and validation purposes
    output_string = "add_random_text"

    # Get length of address string given
    address_length = len(address)
    if verbose: print (f"Inserting random text...")
    if verbose: print (f"Length of input address: {address_length}")

    # Calculate the max number of characters than can be inserted into the address string and still be less than or equal to the max number of characters allowed in the address string. 
    max_insertion_string_length = address_char_lim - address_length
    if verbose: print (f"Max insertion string length: {max_insertion_string_length}")

    # Randomly choose length of random text
    # Adding 1 to the max length to ensure that the random text length can be the max insertion length possible (python-specific issue).
    random_text_length = random.randint(1, max_insertion_string_length + 1)
    if verbose: print (f"Randomly chosen random text length: {random_text_length}")

    # Determine if a separator can be inserted into the random text. If yes, then randomly insert a separator either at the beginning or end of the random text. If there's enough space to add a separator at the beginning AND end of the random text, then do that. Choose separator at random.
    n_allowable_separators = max_insertion_string_length - random_text_length
    n_separators = 2 if n_allowable_separators >= 2 else n_allowable_separators
    separator_options = [" ", "-", "_", ":", ".", ",", "/", "!", "?", "(", ")", "[", "]", "{", "}", "|", "@"]
    if n_separators == 2:
        begin_separator = random.choice(separator_options)
        end_separator = random.choice(separator_options)
    elif n_separators == 1:
        begin_or_end = random.choice([1, 2])
        if begin_or_end == 1:
            begin_separator = random.choice(separator_options)
            end_separator = ""
        else:
            begin_separator = ""
            end_separator = random.choice(separator_options)
    else:
        begin_separator = ""
        end_separator = ""

    if verbose: print (f"Randomly chosen separators: begin_separator: {begin_separator}, end_separator: {end_separator}")

    # Randomly choose an alphanumeric character for each possible character in the random text length.
    random_text = begin_separator
    for c in range(random_text_length):
        random_text_char = random.choice(string.ascii_letters + string.digits)
        random_text += random_text_char
        if verbose: print (f"Randomly chosen random text: {random_text_char}\nRandom text: {random_text}")
    random_text += end_separator
    if verbose: print (f"Randomly chosen random text: {random_text}")

    # Randomly choose a spot in the address to insert the random text
    random_insertion_index = random.randint(0, address_length)
    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Randomly chosen random text: {random_text}")
    if verbose: print (f"Randomly chosen random insertion index: {random_insertion_index}")
    malformed_address = address[:random_insertion_index + 1] + random_text + address[random_insertion_index + 1:]
    if verbose: print (f"Malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the insert_random_words function
# add_random_text_udf = udf(lambda address: add_random_text(address, address_char_lim = 70, verbose = True), ArrayType(StringType())) 

add_random_text_udf = spark.udf.register("add_random_text_udf", lambda address: add_random_text(address, address_char_lim = 70, verbose = True), ArrayType(StringType())) 

# COMMAND ----------


# Convert cardinal directions

def convert_cardinal_directions(address, verbose = True):

    """
    convert_cardinal_directions will take an address and detect if there are cardinal directions within it. Examples of this are "West", "East", "North" and "South". It will then take those cardinal directions and convert them to other forms of the same word. For example, "West" might become "W" or "Wst". The fucntion will return the malformed address with the cardinal directions converted.

    Params:
    address (string): The address string to be modified.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True. 
    """

    import random

    # Instantiate output string
    output_string = "convert_cardinal_directions"

    cardinal_directions_in_address = {"North":[], "South":[], "East":[], "West":[]}
    cardinal_direction_substitutes = {"North":[" North ", " N ", " Nrth ", " N. "], "South":[" South ", " S ", " Sth ", " S. "], "East":[" East ", " E ", " Est ", " E. "], "West":[" West ", " W ", " Wst ", " W. "]}

    # North
    if ("North" in address): cardinal_directions_in_address['North'].append(" North ")
    if (" N " in address): cardinal_directions_in_address['North'].append(" N ")
    if (" Nrth " in address): cardinal_directions_in_address['North'].append(" Nrth ")
    if (" N. " in address): cardinal_directions_in_address['North'].append(" N. ")

    # South
    if ("South" in address): cardinal_directions_in_address['South'].append(" South ")
    if (" S " in address): cardinal_directions_in_address['South'].append(" S ")
    if ("Sth" in address): cardinal_directions_in_address['South'].append(" Sth ")
    if (" S. " in address): cardinal_directions_in_address['South'].append(" S. ")

    # East
    if ("East" in address): cardinal_directions_in_address['East'].append(" East ")
    if (" E " in address): cardinal_directions_in_address['East'].append(" E ")
    # if ("est" in address.lower()): cardinal_directions_in_address['East'].append(" est ") # Commenting because "est" is to common in addresses to be subbing out.
    if (" E. " in address): cardinal_directions_in_address['East'].append(" E. ")

    # West
    if ("West" in address): cardinal_directions_in_address['West'].append(" West ")
    if (" W " in address): cardinal_directions_in_address['West'].append(" W ")
    if ("Wst" in address): cardinal_directions_in_address['West'].append(" Wst ")
    if (" W. " in address): cardinal_directions_in_address['West'].append(" W. ")

    if verbose: print (f"Cardinal directions found in address: {cardinal_directions_in_address}")

    malformed_address = address # Instantiating malformed address so it can be iteratively malformed
    for k, v in cardinal_directions_in_address.items():
        if len(v) > 0:
            print (f"Cardinal direction(s) found in address: {k}: {v}")
            substitutions = []

            for i in range(len(v)):
                sublist = cardinal_direction_substitutes[k] # Copying list to avoid modifying original.
                print (f"Sublist: {sublist}")
                print (f"Removing {v[i]} from sublist")
                sublist.remove(v[i]) # Dropping v[i] so a card direction is not substituted for itself.
                print (f"Sublist: {sublist}")
                substitute_cardinal_direction = random.choice(sublist)
                if verbose: print (f"String to be substituted out: {v[i]}")
                if verbose: print (f"String to be substituted in: {substitute_cardinal_direction}")    
                malformed_address = malformed_address.replace(v[i], substitute_cardinal_direction, 1)
                print (f"Malformed address: {malformed_address}")
        else:
            pass

    print (f"Final malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the insert_random_words function
# convert_cardinal_directions_udf = udf(lambda address: convert_cardinal_directions(address, verbose = True), ArrayType(StringType())) 

convert_cardinal_directions_udf = spark.udf.register("convert_cardinal_directions_udf", lambda address: convert_cardinal_directions(address, verbose = True), ArrayType(StringType())) 

# COMMAND ----------

# Change case

def change_case(address, max_proportion = .5, verbose = True):

    """
    change_case will take an address and randomly change the case of alpha characters within. The randomly-chosen alpha characters, if upper case in the original address, will be changed to lower case and vice versa. 

    Params:
    address (string): The address string to be modified.
    max_proportion (float): The maximum proportion of characters to be changed. Default is .5. The proportion of characters that will be changed will be determined randomly.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True. 
    """

    import random

    # For testing and validation purposes
    output_string = "change_case"

    if verbose: print (f"Original address: {address}")

    # Identify all characters, whether they are alpha or not and their respective case
    alpha_char_indices_dct = {}
    for ind, ltr in enumerate(address):
        if ltr.isalpha():
            alpha_char_indices_dct[ind] = (ltr, ltr.isalpha(), ltr.islower())

    if verbose: print (f"N characters in address: {len(address)}\nN alpha characters in address: {len(alpha_char_indices_dct)}")

    # Randomly choose proportion of alpha characters that will be changed
    change_proportion = round(random.uniform(0, max_proportion), 3)
    if verbose: print (f"Proportion of characters to be changed: {change_proportion}")
    n_alpha_chars_to_change_case = int(change_proportion * len(alpha_char_indices_dct))
    
    # To make training effective, make sure at least one character is changed.
    if n_alpha_chars_to_change_case == 0: n_alpha_chars_to_change_case = 1
    if verbose: print (f"N alpha chars to change case: {n_alpha_chars_to_change_case}")

    # Randomly choose which characters will have their case changed.
    address_char_change_indices = random.sample(list(alpha_char_indices_dct.keys()), n_alpha_chars_to_change_case)
    if verbose: print (f"Randomly-chosen indices to be changed: {address_char_change_indices}")
    if verbose: print (f"Randomly-chosen characters to be changed: {[address[ind] for ind in address_char_change_indices]}")

    # Change randomly-chosen characters' case according to alpha_char_indices_dct
    malformed_address = list(address) # Convert address to list to support character replacement

    for ind in address_char_change_indices:
        if verbose: print (f"Index to be changed: {ind}: {malformed_address[ind]}")
        if alpha_char_indices_dct[ind][2] == True:
            if verbose: print (f"Character changed from: {malformed_address[ind]}, to {malformed_address[ind].upper()}")
            malformed_address[ind] = malformed_address[ind].upper()
        else:
            if verbose: print (f"Character changed from: {malformed_address[ind]}, to {malformed_address[ind].lower()}")
            malformed_address[ind] = malformed_address[ind].lower()
        print (f"New malformed address: {''.join(malformed_address)}")

    malformed_address = ''.join(malformed_address)
    if verbose: print (f"Final malformed address: {malformed_address}")
    
    return (malformed_address, output_string)

# Create a UDF for the insert_random_words function
# change_case_udf = udf(lambda address: change_case(address, max_proportion = 1, verbose = True), ArrayType(StringType()))   

change_case_udf = spark.udf.register("change_case_udf", lambda address: change_case(address, max_proportion = 1, verbose = True), ArrayType(StringType()))

# COMMAND ----------


def add_special_characters(address, max_address_len = 70, max_n_spec_char_prop = .1, verbose = True):

    """
    add_special_characters will take an address and randomly insert special characters into the address. The special characters will be randomly chosen from a list of special characters.

    Params:
    address (string): The address string to be modified.
    max_address_len (int): The maximum number of strings allowed in the address. This ensures that the number of special characters inserted into the address doesn't cause the final malformed address to be too long. Default is 70.
    max_n_spec_char_prop (float): The maximum proportion of characters to be changed. This ensures that the address doesn't become too heavy in special characters. Default is 10%.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    import random

    # Output string to be used for testing and validating purposes
    output_string = "add_special_characters"
    
    special_characters = [" ", "-", "_", ":", ".", ",", "/", "!", "?", "(", ")", "[", "]", "{", "}", "|", "@"]

    # Calculate possible number of special characters that can be inserted into the address
    # If number of possible special characters is less than the max_n_spec_char_prop, set n_possible_special_characters to max_n_spec_char_prop
    if max_address_len - len(address) < int(max_n_spec_char_prop * len(address)):
        n_possible_special_characters = int(max_n_spec_char_prop * len(address))
    else:
        n_possible_special_characters = max_address_len - len(address)
    
    if verbose: print (f"Number of possible special characters: {n_possible_special_characters}")


    # Randomly choose number of special characters to be inserted into address string
    n_special_characters = random.randint(1, n_possible_special_characters)
    if verbose: print (f"Number of special characters to be inserted: {n_special_characters}")

    # Randomly insert special characters into address string
    malformed_address = address
    for i in range(n_special_characters):
        random_insertion_character = random.choice(special_characters)
        random_insertion_index = random.randint(0, len(malformed_address))
        malformed_address = malformed_address[:random_insertion_index] + random_insertion_character + malformed_address[random_insertion_index:]
        
        if verbose: print (f"Updated malformed address: {malformed_address}")

    if verbose: print (f"Final malformed address: {malformed_address}")

    return (malformed_address, output_string)
        

# Create a UDF for the insert_random_words function
# add_special_characters_udf = udf(lambda address: add_special_characters(address), ArrayType(StringType())) 

add_special_characters_udf = spark.udf.register("add_special_characters_udf", lambda address: add_special_characters(address, verbose = True), ArrayType(StringType()))

# COMMAND ----------

# DELETE
k, o = add_special_characters("2124 Wildflower Ct.", max_address_len = 70, max_n_spec_char_prop = .1, verbose = True)
print(k)

# Convert to spaces, then reduce multiple spaces to single, then predict.

# COMMAND ----------

def change_ordinal_numbers(address, ordinal_number_dct = ordinal_numbers_dct, verbose = True):
    """
    change_ordinal_numbers will take an address and change the ordinal numbers in the address. For example, 1st might be changed to First, 42nd will be changed to forty-second or forty 2nd and so on.

    Params:
    address (string): The address string to be modified.
    ordinal_number_dct (dict): Dictionary of ordinal numbers and their string replacements.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    import random

    # Output string to be used for testing and validating purposes
    output_string = 'change_ordinal_numbers'

    # Copies address to create a malformed address that can be iteratively malformed without altering the original
    malformed_address = address

    # Iterates through all elements in ordinal_number_dct and identifies those that are found in the address and possible ordinal numbers for string replacement. These are separated into a dictionary of ordinal numbers and their possible string replacements. Using a dictionary makes it easier to match the ordinal number to it's string when replacing them in the original address.
    possible_ordinal_matches = {}
    for ordinal_number in reversed(sorted(ordinal_number_dct.keys())):
        if ordinal_number.upper() in address.upper():
            if verbose: print (f"Ordinal number found: {ordinal_number}, Ordinal number string: {ordinal_number_dct[ordinal_number]}")
            possible_ordinal_matches[ordinal_number] = ordinal_number_dct[ordinal_number]
            print (f"Possible ordinal matches: {possible_ordinal_matches}")
    
    if possible_ordinal_matches != {}:
        # In many cases, there are more than one ordinal number malform patterns. For example, "21st Ave" could become "Twenty First Ave" or it could become "Twenty 1st Ave". We want all of these options represented in the malformed address dataset, so we are going to randomly choose these options. 

        # This method has some caveats. For example, the "teen" numbers, the one with the longest length is chosen. This is because with a street like 18th street, it will trigger not just "18th" but also "8th". We want the longest one because it is the most complete version of the ordinal number. The same is true for things like 285th Street where 85th is identified as well as 285th and 5th. So this ensures a match to the actual (longest) ordinal number.
        ordinal_match = (max(possible_ordinal_matches.keys() , key = len))
        if verbose: print (f"Ordinal match: {ordinal_match}")

        malformed_address = malformed_address.replace(ordinal_match.upper(), random.choice(possible_ordinal_matches[ordinal_match]).upper())
        if verbose: print (f"Updated malformed address: {malformed_address}")
        
    if possible_ordinal_matches == {}: print (f"No oridnal numbers found in address: {address}")

    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Final malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the change_ordinal_numbers function
# change_ordinal_numbers_udf = udf(lambda address: change_ordinal_numbers(address, ordinal_number_dct = ordinal_numbers_dct, verbose = True), ArrayType(StringType())) 

change_ordinal_numbers_udf = spark.udf.register("change_ordinal_numbers_udf", lambda address: change_ordinal_numbers(address, ordinal_number_dct = ordinal_numbers_dct, verbose = True), ArrayType(StringType()))

# COMMAND ----------

# Included in Utilities noteobok. No longer needed here.

# def generate_cardinal_direction_malforms(norths = [' n ', ' north ', ' nth ', ' nrth ', ' noth '], souths = [' s ', ' south ', ' sth ', ' soth '], easts = [' e ', ' east ', ' est '], wests = [' w ', ' west ', ' wst '], seps = [" ", ""], verbose = True):
    
#     """
#     generate_cardinal_direction_malforms will take various versions of cardinal directions and generate combinations that can be used in downstream address malformations.

#     Params:
#         norths (list of strings): List of various versions of the representation of the North cardinal direction.
#         souths (list of strings): List of various versions of the representation of the South cardinal direction.
#         easts (list of strings): List of various versions of the representation of the East cardinal direction.
#         wests (list of strings): List of various versions of the representation of the West cardinal direction.
#         verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
#     """

#     # Print number of inputs
#     if verbose:
#         print (f"Number of North inputs: {len(norths)}")
#         print (f"Number of South inputs: {len(souths)}")
#         print (f"Number of East inputs: {len(easts)}")
#         print (f"Number of West inputs: {len(wests)}")

#     # Create all possible variations of cardinal directions that contain a version of "North", including things like "Northwest", "Northeast", etc.
#     northwests = []
#     northeasts = []
#     southwests = []
#     southeasts = []

#     for ns in [norths, souths]:
#         for nsi in ns:
#             for sep in seps:
#                 for ew in [easts, wests]:
#                     for ewi in ew:
#                         n_add = " " + nsi + sep + ewi + " "

#                         # Add north wests
#                         # Removes double and triple spaces that are introduced when joining norths/souths to easts/wests.
#                         if ("n" in nsi) & ("w" in ewi):
#                             northwests.append(n_add.replace("   ", " ").replace("  ", " "))
#                             northwests.append(" " + n_add.replace(" ", "") + " ")

#                         # Add north easts
#                         # Removes double and triple spaces that are introduced when joining norths/souths to easts/wests.
#                         elif ("n" in nsi) & ("w" not in ewi):
#                             northeasts.append(n_add.replace("   ", " ").replace("  ", " "))
#                             northeasts.append(" " + n_add.replace(" ", "") + " ")

#                         # Add south wests
#                         # Removes double and triple spaces that are introduced when joining norths/souths to easts/wests.
#                         elif ("s" in nsi) & ("w" in ewi):
#                             southwests.append(n_add.replace("   ", " ").replace("  ", " "))
#                             southwests.append(" " + n_add.replace(" ", "") + " ")

#                         # Add north easts
#                         # Removes double and triple spaces that are introduced when joining norths/souths to easts/wests.
#                         elif ("s" in nsi) & ("w" not in ewi):
#                             southeasts.append(n_add.replace("   ", " ").replace("  ", " "))
#                             southeasts.append(" " + n_add.replace(" ", "") + " ")
    
#     # Deduplicates each list
#     norths = list(set(norths))
#     souths = list(set(souths))
#     easts = list(set(easts))
#     wests = list(set(wests))
#     northwests = list(set(northwests))
#     northeasts = list(set(northeasts))
#     southwests = list(set(southwests))
#     southeasts = list(set(southeasts))

#     # Print details of results
#     if verbose: 
#         print(f"Number of North variations: {len(norths)}")
#         print(f"Number of South variations: {len(souths)}")
#         print(f"Number of East variations: {len(easts)}")
#         print(f"Number of West variations: {len(wests)}")
#         print(f"Number of North West variations: {len(northwests)}")
#         print(f"Number of North East variations: {len(northeasts)}")
#         print(f"Number of South West variations: {len(southwests)}")
#         print(f"Number of South East variations: {len(southeasts)}")

#     return norths, souths, easts, wests, northwests, northeasts, southwests, southeasts

# COMMAND ----------

# Included in Utilities noteobok. No longer needed here.

# norths, souths, easts, wests, northwests, northeasts, southwests, southeasts = generate_cardinal_direction_malforms()
# cardinal_direction_lists = [norths, souths, easts, wests, northwests, northeasts, southwests, southeasts]

# easts

# COMMAND ----------

# V2
def change_cardinal_directions(address, verbose = True):
    
    """
    change_cardinal_directions will take an address, scan it for cardinal directions ("North", "South", "East", "West") and randomly change them to other versions. For example "2125 Sunflower Ct. Northeast" might become "2125 Sunflower Ct. NE" or "2125 Sunflower Ct. Nrth est".

    Params:
    address (string): The address string to be modified.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """
    
    import random

    # Output string to be used for testing and validating purposes
    output_string = "change_cardinal_directions"

    # Print address given for visibility
    if verbose: print (f"Original address: {address}")

    # Instantiate malformed address to be used for malform process while maintaining original address for later comparison
    malformed_address = address

    # Generate all possible, readable versions of cardinal directions
    # Combine norths and souths into one list for ease of process
    norths, souths, easts, wests, northwests, northeasts, southwests, southeasts = generate_cardinal_direction_malforms()
    cardinal_direction_lists = [norths, souths, easts, wests, northwests, northeasts, southwests, southeasts]

    # Loop through each possible cardinal direction to see if it's present in the address given
    cardinal_directions_found = {}
    ending_cardinal_direction_dct = {}

    for cdl in cardinal_direction_lists:
        for cd in cdl:

            # Using .upper() method in order to eliminate the need to identify all possible combinations of upper and lower case cardinal directions. The addresses from Smarty are all upper case, so the code here is built to match that. It's just easier to just have everything in the same case.
            cardinal_direction_present_n = address.count(cd)
            ends_w_cardinal_direction = (address.endswith(cd.rstrip()))

            if cardinal_direction_present_n > 0:
                if verbose: print (f"Cardinal direction {cd} present {cardinal_direction_present_n} times in address: {address}")                

                # Randomly choose another version of the identified cardinal direction for every instance that cardinal direction shows up in the original address
                for i in range(cardinal_direction_present_n):
                    cd_malformed = random.choice([cdli for cdli in cdl if cdli != cd])
                    cardinal_directions_found.setdefault(cd, []).append(cd_malformed)
                    if verbose: print (f"Substitution cardinal direction randomly chosen: {cd}...")

            # Add ending cardinal direction to dictionary with key = cardinal direction, value is a tuple that includes the length of the cardinal direction and a randomly-chosen malformed cardinal direction. A space at the end is included randomly. This will be used later to replace the cardinal direction at the end of the address. 
            if ends_w_cardinal_direction:
                # Identify the ending cardinal direction for replacement later
                ending_cardinal_direction = cd.rstrip()
                if verbose: print (f"{address} with cardinal direction {ending_cardinal_direction}")

                # Calculate the length of the ending cardinal direction so it can be replaced with a malformed version without impacting occurrences in the address string that are NOT at the end of the address string. The reason this is important is that there will be cases where an address might have "north" in it that is not at the end of the address string. For example, "2124 Northeast Sunflower Court North". In this case, we only want to malform the ending "north", while preserving the "northeast" so it can be treated as a separate cardinal direction.
                len_ending_cardinal_direction = len(cd.rstrip())
                if verbose: print (f"Length of ending cardinal direction: {len_ending_cardinal_direction}")

                # Add details of ending cardinal direction to dictionary
                ending_cardinal_direction_dct[ending_cardinal_direction] = (len_ending_cardinal_direction,)
                cd_malformed = random.choice([cdli for cdli in cdl if cdli != cd]).rstrip()
                add_ending_space = random.choice(["", " "])
                if verbose: print (f"Malformed ending cardinal direction: '{cd_malformed + add_ending_space}'")
                ending_cardinal_direction_dct[ending_cardinal_direction] = ending_cardinal_direction_dct[ending_cardinal_direction] + (cd_malformed, add_ending_space)
    
    # The cardinal_directions_found dictionary contains a list entry for each cardinal direction found. The list for each cardinal direction contains a cardinal direction substitute (malformed string) for each time the respective cardinal direction occurrs in the original address provided. This part of the function will substitute each occurrence with the respective substitute (malformed string).
    print (f"Cardinal directions dictionary: {cardinal_directions_found}")
    print (f"Ending cardinal direction dictionary: {ending_cardinal_direction_dct}")

    for cd, cd_repl_lst in cardinal_directions_found.items():
        if verbose: print (f"Cardinal direction: {cd}, list: {cardinal_directions_found[cd]}")
        # if verbose: print (f"Cardinal direction: {cd}")
        for i in range(len(cd_repl_lst)):
            if verbose: print (f"Cardinal direction replacement: {cd_repl_lst[i]}")
            malformed_address = malformed_address.replace(cd, cd_repl_lst[i], 1)
            if verbose: print (f"Malformed address: {malformed_address}")

    # Replace ending cardinal direction with malformed version by replacing the length of the ending cardinal direction with the malformed version.
    if ending_cardinal_direction_dct:
        if verbose: print (f"Address has ending cardinal direction: {ending_cardinal_direction_dct}")
        for cd in ending_cardinal_direction_dct:
            print (f"Ending cardinal directions: {cd}")
            end_card_dir_start_index = ending_cardinal_direction_dct[cd][0]
            end_card_dir_rplc_str = ending_cardinal_direction_dct[cd][1]
            malformed_address = malformed_address[:-end_card_dir_start_index] + " " + end_card_dir_rplc_str
            if verbose: print (f"Malformed address: {malformed_address}")
    
    print (f"Original address: {address}")
    print (f"Final malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the change_cardinal_directions function
# change_cardinal_directions_udf = udf(lambda address: change_cardinal_directions(address, verbose = True), ArrayType(StringType())) 

change_cardinal_directions_udf = spark.udf.register("change_cardinal_directions_udf", lambda address: change_cardinal_directions(address, verbose = True), ArrayType(StringType()))

# COMMAND ----------

def insert_random_cardinal_directions(address, verbose = True):
    
    """
    insert_random_cardinal_directions will take an address and randomly insert cardinal directions. For example "2124 Sunflower Ct." might become "2124 NE Sunflower Ct." or "2124 North Sunflower Ct.", or even "2124 Nrth. In the event that cardinal directions are already in the address provided, it will attempt to find them and not add anymore.
    
    Params:
    address (string): The address string to be modified.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    import random
    import re

    # Output string to be used for testing and validating purposes
    output_string = "insert_random_cardinal_directions"

    # Generate all possible, readable versions of cardinal directions
    # Combine norths and souths into one list for ease of process
    norths, souths, easts, wests, northwests, northeasts, southwests, southeasts = generate_cardinal_direction_malforms()
    cardinal_direction_list = []
    for l in [norths, souths, easts, wests, northwests, northeasts, southwests, southeasts]:
        cardinal_direction_list.extend(l)

    # Looks for cardinal directions in the address. If there are cardinal directions already present, the function will stop.
    for cd in cardinal_direction_list:

        # Using .lower() method in order to eliminate the need to identify all possible combinations of upper and lower case cardinal directions. It's just easier to just lowercase everything.
        cardinal_direction_present_n = address.lower().count(cd)
        ends_w_cardinal_direction = (address.lower().endswith(cd.rstrip()))

        if (cardinal_direction_present_n > 0) | (ends_w_cardinal_direction):
            if verbose: print (f"Cardinal direction {cd} present {cardinal_direction_present_n} times in address: {address}") 
            if verbose: print (f"Cardinal directions already present in address.")

            return address

    # In the US, most addresses start with numerical characters. This function will not consider inserting cardinal directions before or between these initial numerical characters. To facilitate this, the code below will identify the index of the first non-numeric character and only consider inserting cardinal directions at or after this index.
    first_non_num = -1
    while first_non_num < 0:
        for i in range(len(address)):
            if verbose: print (address[i], address[i].isdigit())
            if address[i].isdigit():
                continue
            else:
                first_non_num = i
                break
    if verbose: print (f"First non-numeric character: {address[first_non_num]} at index: {first_non_num}")

    #Find all instances of spaces in address. These are good spots to insert random cardinal directions. Also including the end of the address, where there might not be a space but would still make for a good spot to randomly insert a cardinal direction.
    space_indices = [m.start() for m in re.finditer(' ', address)] 
    if address[-1] != ' ': space_indices.append(-1)
    if verbose: print (f"Space indices: {space_indices}")

    # Randomly choose the number of cardinal directions to insert 
    num_card_dir_inserts = random.randint(1, len(space_indices))
    if verbose: print (f"Number of cardinal directions to insert: {num_card_dir_inserts}")

    # Randomly choose the cardinal directions to insert
    card_dir_inserts = []
    while len(card_dir_inserts) < num_card_dir_inserts:
        cdi = random.choice(cardinal_direction_list)
        card_dir_inserts.append(cdi)
    if verbose: print (f"Cardinal directions to insert: {card_dir_inserts}")
        
    # Randomly choose which indices to insert the cardinal directions.
    # Indices should not be chosen twice
    # Index list is sorted to ensure effective use of adj_factor later in the fuction
    indices_to_insert = []
    while (len(indices_to_insert) < num_card_dir_inserts) & (len(indices_to_insert) < len(space_indices)):
        i = random.choice(space_indices)
        print (f"i: {i}")
        if i not in indices_to_insert:
            indices_to_insert.append(i)
    indices_to_insert.sort()

    # Create copy of original address to iteratively malform without altering original address.
    malformed_address = address
    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Malformed address: {malformed_address}")

    # Care needs to be taken to ensure that indices where cardinal directions are inserted are adjusted as cardinal directions are inserted. For example, the address "2124 Sunflower Ct." has 2 spaces; one at index 4 and the other at index 14. If we insert " North" at index 4, then the address becomes "2124 North Sunflower Ct." and the space that was previously at index 14 is now at index 20. So we must adjust the indices_to_insert list accordingly. We use the adj_factor variable to account for this. It is simply a cumulation of the lenghts of the cardinal directions that have been inserted.
    adj_factor = 0
    for i in range(num_card_dir_inserts):
        # Get the first index to be used to insert cardinal direction
        insert_index = indices_to_insert[i]
        if verbose: print (f"Insert index: {insert_index}")

        # Get the cardinal direction to be inserted
        card_dir_insert = card_dir_inserts[i]
        if verbose: print (f"Cardinal direction to insert: {card_dir_insert}")
        
        # Removes trailing space from inserted card dir if there is one. This eliminates possibility of resulting double spaces.
        if card_dir_insert[-1] == " ":
            card_dir_insert = card_dir_insert[:-1]
            if verbose: print (f"Space removed after inserted cardinal direction: {card_dir_insert}")

        # Insert cardinal direction
        # Since the last index ([-1]) is not impacted by the adjustment factor, we can ignore the adjustment factor when inserting into this index.
        if insert_index == -1:
            malformed_address = malformed_address + card_dir_insert
            if verbose: print (f"Malformed address: {malformed_address}")
        else:
            malformed_address = malformed_address[:insert_index + adj_factor] + card_dir_insert + malformed_address[insert_index + adj_factor:]
            if verbose: print (f"Malformed address: {malformed_address}")

            card_dir_len = len(card_dir_insert)
            adj_factor += card_dir_len
            if verbose: print (f"Adjustment factor: {adj_factor}")

    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the insert_random_cardinal_directions function
# insert_random_cardinal_directions_udf = udf(lambda address: insert_random_cardinal_directions(address, verbose = True), ArrayType(StringType())) 

insert_random_cardinal_directions_udf = spark.udf.register("insert_random_cardinal_directions_udf", lambda address: insert_random_cardinal_directions(address, verbose = True), ArrayType(StringType()))

# COMMAND ----------





# COMMAND ----------

def randomly_remove_vowels(address, verbose = True):

    """randomly_remove_vowels will take an address and randomly remove vowels. This will allow the address to still be readable, but still be fundamentally different from it's original text.

    Params:
    address (string): The address string to be modified.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    import random

    # Output string to be used for testing and validating purposes
    output_string = "randomly_replace_vowels"

    # Print address given for reference
    if verbose: print (f"Original address: {address}")

    # Converts address to lowercase to reduce the number of characters searched for (no longer have to search for upper and lower case characters)
    lowercase_address = address.lower()
    if verbose: print (f"Lowercase address: {lowercase_address}")

    # Finds indices of all vowels in address
    vowel_list = ['a', 'e', 'i', 'o', 'u']
    vowel_indices = [i for i, char in enumerate(lowercase_address) if char in vowel_list]
    if verbose: print (f"N vowels in address: {len(vowel_indices)}, Vowel indices: {vowel_indices}")
                    
    # Randomly choose the number of vowels to remove from address
    try:
        num_vowels_to_remove = random.randint(1, len(vowel_indices))
    except:
        num_vowels_to_remove = 0
    if verbose: print (f"Number of vowels to remove: {num_vowels_to_remove}")

    # Randomly choose which indices to remove vowels from
    indices_to_remove = random.sample(vowel_indices, num_vowels_to_remove)
    indices_to_remove.sort()
    if verbose: print (f"Indices randomly chosen for removal: {indices_to_remove}")

    # Copy original address as a separate string to be malformed while retaining original address provided to function
    malformed_address = address

    # Remove randomly-chosen vowels
    # Need to keep track of how many vowels have been removed with each iteration so as to adjust the index of the next vowel to remove
    adj_factor = 0
    for i in indices_to_remove:
        if verbose: print (f"Vowel index to be removed: {i - adj_factor}")
        if verbose: print (f"Vowel to be removed: {malformed_address[i - adj_factor]}")
        malformed_address = malformed_address[:i - adj_factor] + malformed_address[i + 1 - adj_factor:]
        adj_factor += 1
        if verbose: print (f"Malformed address: {malformed_address}")
                        
    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Final malformed address: {malformed_address}")

    return (malformed_address, output_string)

# Create a UDF for the randomly_remove_vowels function
# randomly_remove_vowels_udf = udf(lambda address: randomly_remove_vowels(address, verbose = True), ArrayType(StringType())) 

randomly_remove_vowels_udf = spark.udf.register("randomly_remove_vowels", lambda address: randomly_remove_vowels(address, verbose = True), ArrayType(StringType()))

# COMMAND ----------

def randomly_replace_spaces(address, replace_list = ["`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "+", "=", "-", "{", "}", "|", ":", "<", ">", "?", "/", ".", ",", "'", ";", "\\", "]", "[", "]", " "], verbose = True):

    """randomly_replace_spaces will take an address and randomly replace spaces with special characters or remove the space entirely. This will allow the address to still be readable, but still be fundamentally different from it's original text.

    Params:
    address (string): The address string to be modified.
    replace_list (list): List of special characters to be used to replace spaces. Default list provided
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    # Output string to be used for testing and validating purposes
    output_string = "randomly_replace_spaces"

    import random

    # Print address given for reference
    if verbose: print (f"Original address: {address}")

    # Finds indices of all spaces
    space_indices = [i for i, char in enumerate(address) if char == " "]
    if verbose: print (f"N spaces in address: {len(space_indices)}, Space indices: {space_indices}")
                    
    # Randomly choose the number of spaces to replace in address
    num_spaces_to_replace = random.randint(1, len(space_indices))
    if verbose: print (f"Number of spaces to replace/remove: {num_spaces_to_replace}")

    # Randomly choose which indices to replace spaces
    indices_to_replace = random.sample(space_indices, num_spaces_to_replace)
    indices_to_replace.sort()
    if verbose: print (f"Indices randomly chosen for replacement/removal: {indices_to_replace}")

    # Copy original address as a separate string to be malformed while retaining original address provided to function
    malformed_address = address

    # Replace randomly-chosen spaces
    # Need to keep track of how many spaces have been removed with each iteration so as to adjust the index of the next vowel to remove
    adj_factor = 0
    for i in indices_to_replace:
        replacement_character = random.choice(replace_list)
        if replacement_character == " ":
            if verbose: print (f"Space index to be REMOVED: {i - adj_factor}")
            malformed_address = malformed_address[:i - adj_factor] + malformed_address[i + 1 - adj_factor:]
            adj_factor += 1
            if verbose: print (f"Malformed address: {malformed_address}")
        else:
            if verbose: print (f"Space index to be replaced: {i - adj_factor}")
            malformed_address = malformed_address[:i - adj_factor] + replacement_character + malformed_address[i - adj_factor + 1:]
            if verbose: print (f"Malformed address: {malformed_address}")
                        
    if verbose: print (f"Original address: {address}")
    if verbose: print (f"Final malformed address: {malformed_address}")
    
    return (malformed_address, output_string)

# Create a UDF for the randomly_remove_vowels function
# randomly_replace_spaces_udf = udf(lambda address: randomly_replace_spaces(address, verbose = True), ArrayType(StringType())) 

randomly_replace_spaces_udf = spark.udf.register("randomly_replace_spaces", lambda address: randomly_replace_spaces(address, verbose = True), ArrayType(StringType()))


# COMMAND ----------

def insert_cardinal_directions(address, verbose = True):
    
    """
    insert_cardinal_directions will take an address, scan it for cardinal directions ("North", "South", "East", "West" and other versions of these) and if none are present, it will randomly insert one. For example "2125 Sunflower Ct." might become "2125 Sunflower Ct. NE" or "2125 West Sunflower Ct.".

    Params:
    address (string): The address string to be modified.
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """
    
    import random

    # Output string to be used for testing and validating purposes
    output_string = "insert_cardinal_directions"

    # Print address given for visibility
    if verbose: print (f"Original address: {address}")

    # Generate all possible, readable versions of cardinal directions
    # Combine norths and souths into one list for ease of process
    norths, souths, easts, wests, northwests, northeasts, southwests, southeasts = generate_cardinal_direction_malforms()
    cardinal_direction_lists = norths + souths + easts + wests + northwests + northeasts + southwests + southeasts

    # Loop through each possible cardinal direction to see if it's present in the address given
    cardinal_directions_found = []

    for cd in cardinal_direction_lists:

        lower_addr = address.lower()

        if cd.lower() in lower_addr:
            cardinal_directions_found.append(cd)
    
    # This function only applies to addresses that don't already have a cardinal direction
    if len(cardinal_directions_found) > 0:
        return address
    
    else:
        
        # Instantiate malformed address to be used for malform process while maintaining original address for later comparison
        malformed_address = address

        # Randomly choose a cardinal direction from the list of cardinal directions
        cd_insert = random.choice(cardinal_direction_lists)

        
        # Randomly choose a space or the beginning or end of the address to insert the cardinal direction
        
        spaces = [i for i, char in enumerate(malformed_address) if char == ' ']
        
        insert_options = ['begin', 'end'] + spaces
        
        insert_option = random.choice(insert_options)

        if insert_option == 'begin':
            malformed_address = cd_insert + ' ' + malformed_address
        elif insert_option == 'end':
            malformed_address = malformed_address + ' ' + cd_insert
        else:
            malformed_address = malformed_address[:insert_option] + ' ' + cd_insert + ' ' + malformed_address[insert_option + 1:]

        return malformed_address

# Create a UDF for the insert_cardinal_directions function
# insert_cardinal_directions_udf = udf(lambda address: insert_cardinal_directions(address, verbose = True), ArrayType(StringType())) 

insert_cardinal_directions_udf = spark.udf.register("insert_cardinal_directions_udf", lambda address: insert_cardinal_directions(address, verbose = True), ArrayType(StringType()))

# COMMAND ----------

def process_data(dataset, function_list, address_column = 'malformed_first_line', verbose = True):

    """process data will take a Pyspark dataframe and a list of function names and will apply each function to the dataset the number of times specified (see description of function_list argument below). The process_data function will return the processed dataset as a Pyspark dataframe. 

    Params:
    dataset (Pyspark dataframe): The name of the column that is to be processed should be called 
    function_list (list): List of function names to be run in the order they are provided in the list. If you want to run a function more than once, simply include it in the list as many times as you want. If you want to run two functions multiple times, but staggered with other functions, simply create the list of functions according to the order you prefer.
    address_column (string): The name of the column that is to be processed. Default is "malformed_first_line".
    verbose (bool): Boolean flag used to print statements showing function progress. Default is True.
    """

    # Add malform_process column to dataset. This allows for downstream malform processes to be added iteratively. 
    output_sdf = dataset.withColumn(address_column + "_malform_steps", lit(""))

    # For purposes of printing progress
    n_functions = len(function_list)
    ind = 1

    # Loop through each function and apply it to the address column.
    for func in function_list:
        print (f"Running function {ind} of {n_functions}: {func.__name__}")

        if verbose: print ()
        # Step 1: Run randomly_replace_spaces function, which returns an array of two outputs (malformed string and description of the malform method)
        output_sdf = output_sdf.withColumn("interim_output", func(col(address_column)))

        # Step 2: In order to separate the returned array into two separate columns, a new sdf is created with the two array elements separated
        output_sdf = output_sdf.select("first_line", "city", "state", "zip", "full_addr", col("interim_output")[0].alias(address_column), concat(col(address_column + "_malform_steps"), lit(", "), col("interim_output")[1]).alias(address_column + "_malform_steps"), "st").drop("interim_output")

    return output_sdf

# COMMAND ----------

# MAGIC %md
# MAGIC ####Build out training set

# COMMAND ----------

# MAGIC %md
# MAGIC ####Automatically generate training dataset

# COMMAND ----------

# NEXT STEPS
# 1. 
# 2. Validate output of each function

# Figure out a way to:
# 1. Randomly choose the number of functions to be applied to a set of addresses
# 2. Apply those functions before appending the data

# Run each function once or twice on the dataset
# Run each combination of 2, 3, 4 functions twice over the dataset
# Get combinations

# COMMAND ----------

def create_combination_lists(function_list, n, perm_or_comb = "combinations", verbose = True):

    """create_combination_lists will take a list of functions and return a list of lists that contains all the combinations (not permutations) of n functions. The purpose of this is to facilitate building out representations of possible ways and orders that addresses can be malformed.
    
    Params:
    fucntion_list (list): List of functions to be used for malforming addresses. The elements of this list are the functions themselves, but string representations of the function names. For example: [replace_char_udf, repeat_characters_udf, repeat_part_of_address_udf]. Notice there are no quotation marks. Again, these are the functions themselves, not the string names of the functions.
    n (integer): The number of functions to be combined. In other words, if n is set to 3 then combinations of 3 functions will be returned.
    perm_or_comb (string): Options are "combinations" or "permutations". This will inform whether combinations or permutations are returned. Default is "combinations".
    """
    
    from itertools import combinations, permutations

    if perm_or_comb == "permutations":
        result = list(permutations(function_list, n))
    else:
        result = list(combinations(function_list, n))
        
    if verbose:        
        print (f"Number of {perm_or_comb} of {n}: {len(result)}")
        print (f"Output lists of len {n}:")
        for c in result:
            print ([l.__name__ for l in c])
    
    return result

# COMMAND ----------

# Create test datframe to serve as basis for new table of malformed addresses which will be appended with training data.

from pyspark.sql.functions import col
test_sdf = spark.sql("""SELECT *, first_line AS malformed_first_line FROM smarty.smarty_samp ORDER BY RAND(42) LIMIT 1000""")

mf_lst = [replace_char_udf]


malformed_sdf = process_data(dataset = test_sdf, function_list = mf_lst, address_column = 'malformed_first_line', verbose = True)

# Create table in newly-created schema (above)

# Get schema string
schema_str = ''
for c in malformed_sdf.columns: 
    if c != 'st':
        print (f"{c} STRING, ")
        schema_str += f"{c} STRING, "  

schema_str = schema_str[:-2]
print (f"\n{schema_str}")

spark.sql(f"""CREATE EXTERNAL TABLE IF NOT EXISTS smarty.smarty_malformed_1000_training (
{schema_str}
)
USING delta
PARTITIONED BY (st STRING)
COMMENT "This table consists of a sample of 1000 records from the original Smarty US address list as obtained via their website: https://www.smarty.com/. Each record has been malformed numerous times in numerous ways. The intention of this dataset is to serve as an ML testing dataset."
LOCATION 's3://jml-address-validation/tables/smarty/training/smarty_malformed_1000_training'""")

# COMMAND ----------

# Populate table with training data
from pyspark.sql import DataFrame

# Build test training set (if not already done above)
test_sdf = spark.sql("""SELECT *, first_line AS malformed_first_line FROM smarty.smarty_samp ORDER BY RAND(42) LIMIT 1000""")

malform_function_list = [replace_char_udf, repeat_characters_udf, repeat_part_of_address_udf, insert_random_words_udf, add_random_text_udf, convert_cardinal_directions_udf, change_case_udf, add_special_characters_udf, change_ordinal_numbers_udf, change_cardinal_directions_udf, insert_random_cardinal_directions_udf, randomly_remove_vowels_udf, randomly_replace_spaces_udf]

n_malform_records_inserted = 0

# Run each malform function X times over the dataset
iterations = 10
for mf in malform_function_list:
    print(mf)
    mf_lst = [mf]

    i = 0
    while i < iterations:
        malformed_sdf = process_data(dataset = test_sdf, function_list = mf_lst, address_column = 'malformed_first_line', verbose = True)
        print (f"    Running {mf} iteration {i} of {iterations} times...")

        # Insert output into table
        malformed_sdf.write.format("delta").mode("append").insertInto("smarty.smarty_malformed_1000_training")
        i += 1
        n_malform_records_inserted += malformed_sdf.count()

        
run_2_perms = False
if run_2_perms == True: 
    # Run each permutation of two functions
    iterations = 5
    function_list_perm_2 = create_combination_lists(malform_function_list, n = 2, verbose = False)
    n_fls = len(function_list_perm_2)

    for fl in function_list_perm_2:
        i = 0
        while i < iterations:
            print (f"    Running {[f.__name__ for f in fl]} iteration {i} of {iterations} times...")
            malformed_sdf = process_data(dataset = test_sdf, function_list = fl, address_column = 'malformed_first_line', verbose = True)

            # Insert output into table
            malformed_sdf.write.format("delta").mode("append").insertInto("smarty.smarty_malformed_1000_training")
            i += 1
            n_malform_records_inserted += malformed_sdf.count()



run_3_perms = False
if run_3_perms == True:
    # Run each permutation of three functions
    iterations = 5
    function_list_perm_3 = create_combination_lists(malform_function_list, n = 3)
    n_fls = len(function_list_perm_3)

    for fl in function_list_perm_3:
        i = 1
        while i < iterations:
            print (f"    Running {[f.__name__ for f in fl]} iteration {i} of {iterations} times...")
            malformed_sdf = process_data(dataset = test_sdf, function_list = fl, address_column = 'malformed_first_line', verbose = True)

            # Insert output into table
            malformed_sdf.write.format("delta").mode("append").insertInto("smarty.smarty_malformed_1000_training")
            i += 1
            n_malform_records_inserted += malformed_sdf.count()

print (" ")
print (f"N malformed records: {n_malform_records_inserted}")
display(malformed_sdf)

# COMMAND ----------

# Insights after first round:
# 1. Probably don't subject digits to repeating character function.
# 2. Review character substitutions to make sure they work. For example the "J"-looking character in this address is supposed to be a V, and while it does work once you're told what it's supposed ot be, it might not be a great substitution: 320OOOO ᦔAAAAคAA̶NCE SͲ

# COMMAND ----------

###############################
#######  DO NOT DELETE  #######
###############################

# PRE-PROCESS BEFORE PREDICTION
# 1. Reduce repeats of characters from sequences of 3+ to just 2
# 2. Remove combining characters using this function:

    import unicodedata

    def remove_combining_characters(input_str):
        if input_str is None:
            return None
        # Normalize to NFD (Normalization Form Decomposed)
        normalized_str = unicodedata.normalize('NFD', input_str)
        
        # Filter out characters that are combining marks
        cleaned_str = ''.join(
            char for char in normalized_str 
            if unicodedata.combining(char) == 0
        )
        return cleaned_str

    cs = remove_combining_characters("1̶̶̶̶̶̶01𝟴͎8͎ W 다섯222222NDDDD Pƚ UNIɬ too08")    
    print (cs)



# COMMAND ----------


