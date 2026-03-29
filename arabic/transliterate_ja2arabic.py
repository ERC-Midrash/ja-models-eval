# ja_transliterator.py
# Character mapping from paper's Appendix A
# Based on the transliteration approach proposed by Moreno Gonzalez et al. (2025): https://arxiv.org/abs/2507.04746
import unicodedata
import re

DOTLESS_MAP = {
    'א': 'ا',  # Alif
    'ב': 'ب',  # Ba
    'ג': 'ج',  # Jim
    'ד': 'د',  # Dal (also ذ)
    'ה': 'ه',  # Ha
    'ו': 'و',  # Waw
    'ז': 'ز',  # Zay
    'ח': 'ح',  # Ha (emphatic)
    'ט': 'ط',  # Ta (emphatic)
    'י': 'ي',  # Ya
    'כ': 'ك', 'ך': 'ك',  # Kaf
    'ל': 'ل',  # Lam
    'מ': 'م', 'ם': 'م',  # Mim
    'נ': 'ن', 'ן': 'ن',  # Nun
    'ס': 'س',  # Sin
    'ע': 'ع',  # Ayin
    'פ': 'ف', 'ף': 'ف',  # Fa
    'צ': 'ص', 'ץ': 'ص',  # Sad
    'ק': 'ق',  # Qaf
    'ר': 'ر',  # Ra
    'ש': 'ش',  # Shin
    'ת': 'ت',  # Ta
}

# Dotted variants (with upper dot diacritic)
DOTTED_MAP = {
    'גׄ': 'غ',  # Ghayn
    'דׄ': 'ذ',  # Dhal
    'הׄ': 'ة',  # Ta marbuta (or ح)
    'טׄ': 'ظ',  # Dha (emphatic)
    'כׄ': 'خ', 'ךׄ': 'خ',  # Kha
    'צׄ': 'ض', 'ץׄ': 'ض',  # Dad
    'תׄ': 'ث',  # Tha
}

def remove_dots(text):
    result = ''
    for char in text:
        if unicodedata.combining(char):
            continue
        result += char
    return result


# Letters that take the upper dot to indicate a distinct Arabic phoneme.
# Includes final forms (ך, ץ).
_DOTTABLE_LETTERS = 'גדהטכךצץת'


def normalize_geresh_to_upper_dot(text):
    """Replace letter + apostrophe with letter + combining upper dot (U+05C4).

    In many digitized JA texts the geresh/apostrophe (') is used instead of the
    Unicode combining upper dot. This function normalises that convention so the
    transliterator's DOTTED_MAP can recognise the dotted letters.

    Only the 7 consonants (+ 2 final forms) that have dotted variants are affected;
    an apostrophe after any other character is left unchanged.
    """
    return re.sub(
        rf'([{_DOTTABLE_LETTERS}])\'',
        '\\1\u05C4',
        text
    )


def transliterate_ja_to_arabic(text, use_dots=True):
    """Transliterate Judeo-Arabic (Hebrew script) to Arabic script"""
    result = []
    i = 0

    if use_dots:
        text = normalize_geresh_to_upper_dot(text)

    if not use_dots:
        text = text.replace("'", '')
        text = remove_dots(text)

    while i < len(text):
        # Check for dotted characters first (2-char sequences)
        if use_dots and i + 1 < len(text):
            two_char = text[i:i + 2]
            if two_char in DOTTED_MAP:
                result.append(DOTTED_MAP[two_char])
                i += 2
                continue

        # Single character mapping
        char = text[i]
        if char in DOTLESS_MAP:
            result.append(DOTLESS_MAP[char])
        else:
            result.append(char)  # Keep punctuation, spaces, etc.
        i += 1

    return ''.join(result)


if __name__ == '__main__':
    ja_word = 'צ̇'
    ja_text = 'אד וכל יציאה פהי איצ̇א הכנסה כמא בינא אקתפא'

    arabic = transliterate_ja_to_arabic(ja_text, use_dots=False)
    print(arabic)

    arabic = transliterate_ja_to_arabic(ja_text, use_dots=True)
    print(arabic)
