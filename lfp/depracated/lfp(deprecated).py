class lfp_data:
    def __init__(self, x, y, file_id, channel, fs, filter=None):
        self.x = x
        self.y = y
        self.file_id = file_id
        self.channel = channel
        self.fs = fs
        self.filter = filter