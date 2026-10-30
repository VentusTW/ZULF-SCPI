
# \<bin\> block embedded header

This document describes the format used by the header embedded in the binary block used by the following SCPI commands implemented by ZULF:

1. Query request: `:AWG:DATA? #<hdr><bin>`
2. Query request: `:AWG:GATing? #<hdr><bin>`
3. Query `:ACQuire:DATA:GET? <index>`  response:  `#<hdr><bin>`

This document doesn't describe the SCPI defined header of the binary block: \#\<hdr\>.
It defines the custom header that is embedded in the \<bin\> body of the requests or responses of the queries listed above.


## \<bin\> - general structure reference
In most cases the \<bin\> structure will use the following structure.

| variable length | variable length | 4B       |
| --------------- | --------------- | -------- |
| Header          | Payload         | Checksum |

The header is positioned in the front of the binary block.


# Header structure

Defines its own version and length.
Carries also the payload version information.
Can provide additional information.

### General layout
	
| Byte 0 \[1B\] | Byte 1 \[1B\] | ... \[ var len\] | Byte n \[var len\] | 
| ------------- | ------------- | ---------------- | ------------------ |
| Fix header    | Fix header    | Extensions       | Extensions         |


#### Fix header

This part defines the length and the actual structure of the header, of its next bytes.
The **fix header** occupies the first only byte of the header. 

| Byte 0 \[1B\]  | Byte 1 \[1B\]   |
| -------------- | --------------- |
| Header version | Payload version |

**Header version** - version of the header. Identifies header length and optional bytes following the fix header bytes (Byte 0 and Byte 1).
The header version is defined externally and independently on the command definition. 

**Payload version** - version of the structure of the payload that comes after the header.
The payload version is defined per command and is unique only in the scope of the command.
Command A and command B can define two different payload definitions identifying them with  the same version number.

#### Further header bytes
Further bytes are dependent on the version of the header.


### Header version =1 
Below structure defines full header in version 1.
It includes also the fix header part that is common for each version of the header.

| Byte 0 \[1B\]  | Byte 1 \[1B\]   | Byte 2 \[1B\] | 
| -------------- | --------------- | ------------- |
| Header version | Payload version | Flow control  |


**Flow control** - 1B-int that is incremented with every next generated chunk of data. If the same block of data is delivered with a repetition of a command or query, then the Flow control part doesn't increment its value.
The recipient can recognise a repetition or discontinuity of a transmission and react accordingly.

### Header version =2 
Below structure defines full header in version 2.
It includes also the fix header part that is common for each version of the header.

| Byte 0 \[1B\]  | Byte 1 \[1B\]   | Bytes \[2-5\] \[4B\] |
| -------------- | --------------- | -------------------- |
| Header version | Payload version | Chunk index          | 


**Chunk index** - 4B-int that can represent individually each chunk of data.
The recipient can recognise a repetition or discontinuity of a transmission and react accordingly.