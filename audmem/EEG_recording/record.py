import pyOpenBCI as bci
import time
import numpy as np
import argparse


def record():
  board = bci.OpenBCICyton(port='/dev/tty.usbserial-DM01MQLD', daisy=True)
  board.write_command('/2')  # record analog read and not accelerometer (could try interleaving later)
  board.write_command('~2')  # Set sample rate (2 for 4000Hz)
  board.write_command('F')  # Initialise SD file for up to 30min recording
  board.write_command('b')  # Start data streaming (i.e. SD writing?)
  finish = False
  while not(finish):
    inp = raw_input("Press 'q' to terminate EEG recording")
    if inp == 'q':
      finish = True
  board.write_command('s')  # Stop data streaming
  board.write_command('j')  # Close SD file


if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('out_file', help=('Path to file where to store EEG data on SD card')
    #args = parser.parse_args()
    record()
