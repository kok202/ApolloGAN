from music21 import converter,instrument
from Util.SimpleFunc import print_debug



def changeToPiano(src, dst):
    try:
        converter_file = converter.parse(src)
        for part in converter_file.parts:
            part.insert(0, instrument.Piano())
        converter_file.write('midi', dst)
    except:
        print_debug("Change to piano Error : May be there is no note in midi")