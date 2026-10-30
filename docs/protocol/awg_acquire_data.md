
# \:AWG\:ACQuire\:DATA\:NEW?

This document defines a response format that is used by the \:AWG\:ACQuire\:DATA:\GET? query.
The response carries the chunk of a requested measurement data.


## General response pattern

The general pattern of the response is:
```
#<hdr><bin>
```



## \<bin\> - general structure
General layout of the \<bin\> part of the DATA command is the following.

| variable length | variable length | 4B       |
| --------------- | --------------- | -------- |
| Header          | Payload         | Checksum |


## Header

The header part is described by the "bin_block_emb_header"" document.

The \:AWG\:GATing? query uses **version 2** of the header.


## Payload

### Payload version =1
Placeholder.

### Payload version =2 

Version 2 of the payload defines a structure visualised in the table below.
The structure repeats within the payload a number of times that is a result of the structure size and applied chunk size.
 
| Element             | Type     | Byte offset | Bit offset | Byte order    | Comment |
| ------------------- | -------- | ----------- | ---------- | ------------- | ------- |
| Channel 1 ADC value | UInt, 2B | 0B          | 0b         | Little endian |         |
| Channel 2 ADC value | UInt, 2B | 2B          | 0b         | Little endian |         |
| Channel 3 ADC value | UInt, 2B | 4B          | 0b         | Little endian |         |
| Channel 4 ADC value | UInt, 2B | 6B          | 0b         | Little endian |         |
| Channel 5 ADC value | UInt, 2B | 8B          | 0b         | Little endian |         |
| Channel 6 ADC value | UInt, 2B | 10B         | 0b         | Little endian |         |
| Channel 7 ADC value | UInt, 2B | 12B         | 0b         | Little endian |         |
| Channel 8 ADC value | UInt, 2B | 14B         | 0b         | Little endian |         |


There is no time (or time offset) reference in the structure. Each two consecutive instances of the structure represent two consecutive physical samples. 


## Return value

The AWG:ACQuire?  query returns "err-\<error code\>" string.
The \<error code\> is 0-255 numerical, decimal value and encodes the following errors:

| error code | meaning             |
| ---------- | ------------------- |
| 0          | no error            |
| 1          | invalid chunk index | 
| 255        | general error       |