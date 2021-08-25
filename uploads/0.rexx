/* PDWPRCEV REXX     Process output of 'runpipe events'              */
/*                   Author: Rob van der Heij, 31 Jul 2017           */

signal on error

'callpipe (end \ name PDWPRCEV.REXX:4)',
   '\ *:',
   '| p: if verify 1 x06 ',
   '| spec 1-* 1 19 2 ',
   '| p: ',
   '| w: pdwfmtev',
   '| *:',
   '\ < pdwfmtev txt | w:'


error: return rc * ( rc <> 12 )

