
# \:AWG\:GATing?

\:AWG\:GATing? \#\<hdr\>\<bin\> is an actual pattern of the command.
Being a query it returns an answer - a result of processing. 

## \<bin\> - general structure
General layout of the \<bin\> part of the GATing command is the following.

| variable length | variable length | 4B       |
| --------------- | --------------- | -------- |
| Header          | Payload         | Checksum |


## Header

The header part is described by the "bin_block_emb_header"" document.

The \:AWG\:GATing? query uses **version 2** of the header.



## Payload

### Payload version =1 
Abandoned.

### Payload version =2

Version 2 of the payload defines a structure visualised in the table below.
The structure repeats within the payload a desired number of times.

Based on the Version 1 the main difference is:
- The bit representation of the \[us\] offset of the EN signals extended to 2B. It lets for over 65\[ms\] of a delay that - in practice - applies to 100 Hz AWG data rate
 
| Element                          | Type     | Byte offset | Bit offset | Byte order    | Comment                                                                                               |
| -------------------------------- | -------- | ----------- | ---------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| Sample delay/offset              | UInt, 2B | 0B          | 0b         | Little endian | Time axis position. Delay / offset /distance from the previous sample expressed in number of samples. |
| CENW: Channel X EN toggle offset | Uint, 2B | 2B          | 0b         | Little endian | The four following elements have same meaning:                                                        |
| CENW: Channel Y EN toggle offset | Uint, 2B | 4B          | 0b         | Little endian | Number of micro-seconds that elapses from the beginning of the given ADC sample period                |
| CENW: Channel Z EN toggle offset | Uint, 2B | 6B          | 0b         | Little endian | to a moment of the corresponding EN signal toggle.                                                    |
| CENW: Channel S EN toggle offset | Uint, 2B | 8B          | 0b         | Little endian | Value of 0xFFFF (65536 dec) means: do not toggle.                                                     |


## Check sum

The checksum field is calculated from all the preceding bytes of the \<bin\> part of the command/query.

The checksum will use the CRC-32 IEEE algoritm, with the 0x04C11DB7 polynom value.
The following details may be worth considering:
> 1. No bit inversions
> 2. 0xFFFFFFFF initial value
> 3. no fianal XOR
> 4. operation on 32bit words

## Return value

The AWG:GATing? query returns "err-\<error code\>" string.
The \<error code\> is 0-255 numerical, decimal value and encodes the following errors:

| error code | meaning                                   |
| ---------- | ----------------------------------------- |
| 0          | no error                                  |
| 1          | repeated block -> block rejected          |
| 2          | block not in order -> block rejected      |
| 3          | header or payload structure not supported |
| 4          | crc mismatch                              |
| 5          | too much data / out of memory             |
| 255        | general error                             |




