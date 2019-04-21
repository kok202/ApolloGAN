import environments as env
from mido import MetaMessage, Message, MidiFile, MidiTrack

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# load
# TrackTimeList = (Track number,   Track Sequence,    pushed sequence time)
# TrackNoteList = (Track number,   Track Sequence,    note, duration for each sequence
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def load(path):
    checkNoteOn         = [0 for i in range(env.MIDI_NOTE_NUMBER)]
    recordNoteStartTime = [0 for i in range(env.MIDI_NOTE_NUMBER)]
    midiFile            = MidiFile(path)
    TrackTimeList       = []
    TrackNoteList       = []

    for i, track in enumerate(midiFile.tracks):
        msgList = []
        msgTime = 0
        for message in track:

            msgTime += message.time
            if message.type == 'note_on':
                cur_note = message.bytes()[1]
                velocity = message.bytes()[2]
                if cur_note != 0:
                    if velocity != 0:
                        checkNoteOn[cur_note] = 1
                        recordNoteStartTime[cur_note] = msgTime
                    elif checkNoteOn[cur_note] == 1:
                        # if velocity == 0, use like note_off
                        checkNoteOn[cur_note] = 0
                        curNoteDuration = (msgTime) - recordNoteStartTime[cur_note]
                        msgList.append([cur_note, recordNoteStartTime[cur_note], curNoteDuration])

            if message.type == 'note_off':
                cur_note = message.bytes()[1]
                if cur_note != 0 and checkNoteOn[cur_note] == 1:
                    checkNoteOn[cur_note] = 0
                    curNoteDuration = (msgTime) - recordNoteStartTime[cur_note]
                    msgList.append([cur_note, recordNoteStartTime[cur_note], curNoteDuration])

        if len(msgList) <= 1:
            continue

        # sort by when note_on is loaded
        def cmpKey(elements):
            return elements[1]
        msgList.sort(key=cmpKey)
        timeBefore = -1
        noteAtTime = []
        noteList = []
        timeList = [0]
        for note, startTime, durationTime in msgList:
            if timeBefore == startTime or startTime == 0:
                # noteAtTime(notes which should be pressed at this timing)
                noteAtTime.append([note, durationTime])
            else:
                noteList.append(noteAtTime)
                noteAtTime = []
                noteAtTime.append([note, durationTime])
                timeList.append(startTime)
                timeBefore = startTime

        # append final note to noteList because it only saved in noteAtTime
        noteList.append(noteAtTime)

        TrackTimeList.append(timeList)
        TrackNoteList.append(noteList)
    return TrackTimeList, TrackNoteList





'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# save
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def save(path, timeList, noteList, metaData=[200000, 'A']):
    def cmpKey(elements):
        return elements[1]
    checkNoteOn       = [0 for i in range(env.MIDI_NOTE_NUMBER)]
    recordNoteOffTime = [-1 for i in range(env.MIDI_NOTE_NUMBER)]
    output_mid = MidiFile()
    track_meta = MidiTrack()
    track_meta.append(MetaMessage('set_tempo', tempo=metaData[0], time=0))
    track_meta.append(MetaMessage('key_signature', key=metaData[1], time=0))
    output_mid.tracks.append(track_meta)

    track_num = len(timeList)
    for i_track in range(track_num):
        track = MidiTrack()
        output_mid.tracks.append(track)
        track.append(Message('program_change', program=12, time=0))
        i_time_before = 0

        # Add enough time 10000 for ending final note off
        finalNote_index = len(timeList[i_track])-1
        if finalNote_index < 0:
            # When note not exist
            break
        finalNote_time = timeList[i_track][finalNote_index]
        finalNote_duration = max(noteList[i_track][finalNote_index], key=cmpKey)[1]+1000

        timeList[i_track].append(finalNote_time + finalNote_duration)
        noteList[i_track].append([[0 ,0]])
        for index, i_time in enumerate(timeList[i_track]):
            # searching note which should be noted off from previous time to next time and add it to list
            NoteOffBetweenTimeGapList = []
            for NoteIndex, NoteOffTime in enumerate(recordNoteOffTime):
                if (i_time_before <= NoteOffTime) & (NoteOffTime <= i_time) & (NoteOffTime != -1):
                    NoteOffBetweenTimeGapList.append([NoteIndex, NoteOffTime])

            # plan : sorting list and note off starting at lowest number
            NoteOffBetweenTimeGapList.sort(key=cmpKey)
            for note, NoteOffBetweenTimeGap in NoteOffBetweenTimeGapList:
                track.append(Message('note_off', note=note, velocity=127, time=NoteOffBetweenTimeGap-i_time_before))
                i_time_before = NoteOffBetweenTimeGap
                checkNoteOn[note] = 0

            # for addiing empty time
            if index != 0:
                track.append(Message('note_on', note=0, velocity=0, time=i_time-i_time_before))
                track.append(Message('note_off', note=0, velocity=0, time=0))
                recordNoteOffTime[0] = -1
            i_time_before = i_time

            # Note on all note which should be played at this time, and update when it should be ended
            for note, duration in noteList[i_track][index]:
                # If it is already set, note off and immediately note on
                if checkNoteOn[note] == 1:
                    track.append(Message('note_off', note=note, velocity=127, time=0))

                track.append(Message('note_on', note=note, velocity=127, time=0))
                recordNoteOffTime[note] = i_time + duration
                checkNoteOn[note] = 1

    output_mid.save(path)



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# sample
# def sample_create_midi
# def sample_using_multihot
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def sample_create_midi(path):
    track1_TimeList =  [0, 960, 1920, 2880]
    track1_NoteList =  [
                            [
                                [62, 960],
                                [66, 960]
                            ],
                            [
                                [64, 960],
                                [68, 960]
                            ],
                            [
                                [66, 960],
                                [70, 960]
                            ],
                            [
                                [68, 960],
                                [72, 960]
                            ]
                        ]

    timeList = []
    noteList = []
    timeList.append(track1_TimeList)
    noteList.append(track1_NoteList)
    save(path, timeList, noteList)



def sample_main():
    sample_create_midi('../{}/Data_TestSet/Sample.mid'.format(env.DATA_INPUT_PATH))