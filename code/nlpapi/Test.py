import Processor as processor

processor.Init()

#from 	Milton, John, Paradise Lost
text_milton = """ Out of our evil seek to bring forth good,
Our labour must be to pervert that end,
And out of good still to find means of evil;
Which ofttimes may succeed so as perhaps
Shall grieve him, if I fail not, and disturb
His inmost counsels from their destined aim.
But see! the angry Victor hath recalled
His ministers of vengeance and pursuit
Back to the gates of Heaven: the sulphurous hail,
Shot after us in storm, o'erblown hath laid
The fiery surge that from the precipice
Of Heaven received us falling; and the thunder,
Winged with red lightning and impetuous rage,
Perhaps hath spent his shafts, and ceases now
To bellow through the vast and boundless Deep.
Let us not slip th' occasion, whether scorn
Or satiate fury yield it from our Foe.
Seest thou yon dreary plain, forlorn and wild,
The seat of desolation, void of light,
Save what the glimmering of these livid flames
Casts pale and dreadful? Thither let us tend
From off the tossing of these fiery waves """

print (processor.getBook(text_milton))