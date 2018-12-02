import os

class Logger(object):
    def __init__(self, logfiledir=None, logfilename=None, printout=False, mute=False):
        print('making logger filename {}, printout:{}'.format(logfilename, printout))
        self.logfiledir = logfiledir
        self.logfilename = logfilename
        self.printout = printout
        if logfiledir==None or logfilename ==None:
           self.printout = True

        self.mute = mute
        if logfiledir is not None:
            os.system("rm {}".format(os.path.join(self.logfiledir, self.logfilename)))

    def log(self, *inputlist):
        if self.printout:
            print(inputlist)
        elif self.mute:
            return
        else:
            inputlist = [str(el) for el in inputlist]
            inputlist = ''.join(inputlist)
            with open(os.path.join(self.logfiledir, self.logfilename), 'a') as f:
                f.write(inputlist + '\n')