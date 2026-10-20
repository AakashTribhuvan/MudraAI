# Trained sign labels

This is the consolidated vocabulary read from the model artifacts currently bundled in `app/backend/models/`.
It lists model labels, not a guarantee that every sign will be recognized correctly in live use. The labels are
kept in their artifact form (`UPPER_SNAKE_CASE`); underscores separate words in a label.

## At a glance

| Recognizer | Distinct labels | Where used |
|---|---:|---|
| Main word sequence model (v8) | 314 | Communication - Words (default) |
| Expanded introduction static-pose model | 32 (10 live-eligible) | Communication - Words - model selector; experimental |
| Fingerspelling alphabet model | 26 | Communication - Spell |
| Phrase model | 99 | Communication - Phrase |

The counts are per model and overlap; do not add them as a count of unique signs. The older v6 word-model
artifact is excluded because v8 is the deployed model. `_BACKGROUND_` in the sequence model means "no sign"
and is not a user-facing sign label.

## Measured accuracy (offline test sets)

| Recognizer | Test result | What the result means |
|---|---|---|
| Main word sequence model (v8) | 98.06% top-1 / 99.93% top-5 (1,495 clips) | Held-out clips, but signers and recording sessions overlap training; per-recording top-1 is about 97.5%. Not a new-signer or live-webcam result. |
| Expanded introduction static-pose model | 17.81% reported test accuracy (73 images) | Exploratory still-image result. Only 10 classes have a live rejection threshold; this does not recognize motion or sentences. |
| Fingerspelling alphabet | 70.93% top-1 / 83.83% top-3 (1,070 photos) | Held-out colour groups; accepted predictions cover 54.3% of samples at 93.8% precision. |
| Phrase model | 58.33% top-1 / 81.25% top-5 (96 videos) | Held-out videos from the same signers; candidates are shown for the user to select. |

These are offline dataset metrics, not a guarantee of camera performance. The static prototype in the
separate `MudraTraining` clone is a different model/evaluation (34.1% top-1 on 88 grouped held-out
images); do not treat the two static-model scores as directly comparable.

## Main word sequence model - 314 labels

Artifact: [`isl_mudra_combined_v8_desk_bilstm/model_card.json`](../app/backend/models/isl_mudra_combined_v8_desk_bilstm/model_card.json).

`ACTOR`, `ADULT`, `AFTERNOON`, `ALIVE`, `ALRIGHT`, `ANIMAL`, `ARTIST`, `ATTACK`, `AUTHOR`,
`BABY`, `BAD`, `BAG`, `BALL`, `BANK`, `BATHROOM`, `BEAR`, `BEAUTIFUL`, `BED`, `BEDROOM`,
`BICYCLE`, `BIG_LARGE`, `BILL`, `BIRD`, `BLACK`, `BLIND`, `BLUE`, `BOAT`, `BOOK`, `BOX`, `BOY`,
`BREAK`, `BRINJAL`, `BROTHER`, `BROWN`, `BUDGET`, `BUS`, `BUSY`, `CABBAGE`, `CAMERA`, `CAR`,
`CARD`, `CARROT`, `CAT`, `CAULIFLOWER`, `CELL_PHONE`, `CHAIR`, `CHEAP`, `CHILD`, `CHILLI`,
`CITY`, `CLEAN`, `CLOCK`, `CLOSE`, `CLOTHING`, `COLD`, `COLOUR`, `COME`, `COMPUTER`, `COOK`,
`COOL`, `COURT`, `COW`, `CROCODILE`, `CROWD`, `CRY`, `CUCUMBER`, `CURVED`, `DAUGHTER`, `DEAD`,
`DEAF`, `DEATH`, `DEEP`, `DEER`, `DIRTY`, `DOCTOR`, `DOG`, `DOOR`, `DREAM`, `DRESS`, `DRINK`,
`DRY`, `ELECTION`, `ELEPHANT`, `ENERGY`, `EVENING`, `EXAM`, `EXERCISE`, `EXPENSIVE`,
`EX_MONSOON`, `FALL`, `FAMILY`, `FAMOUS`, `FAN`, `FAST`, `FATHER`, `FEDUP`, `FEMALE`, `FEVER`,
`FISH`, `FLAT`, `FRIDAY`, `FRIEND`, `GIFT`, `GIRAFFE`, `GIRL`, `GIVE`, `GOD`, `GOOD`,
`GOOD_AFTERNOON`, `GOOD_EVENING`, `GOOD_MORNING`, `GOOD_NIGHT`, `GRANDFATHER`, `GRANDMOTHER`,
`GREEN`, `GREY`, `GROUND`, `GUN`, `HAPPY`, `HARD`, `HAT`, `HE`, `HEALTHY`, `HEAVY`, `HELLO`,
`HIGH`, `HORSE`, `HOSPITAL`, `HOT`, `HOUR`, `HOUSE`, `HOW_ARE_YOU`, `HUG`, `HUSBAND`, `I`,
`INDIA`, `INJURY`, `INTERVIEW`, `IT`, `JOB`, `JUMP`, `KARNATAKA`, `KEY`, `KING`, `KITCHEN`,
`KNIFE`, `LAMP`, `LAPTOP`, `LAWYER`, `LEMON`, `LETTER`, `LIBRARY`, `LIGHT`, `LION`, `LOCATION`,
`LOCK`, `LONG`, `LOOSE`, `LOUD`, `LOW`, `MALE`, `MAN`, `MANAGER`, `MARKET`, `MARRIAGE`, `MATHS`,
`MAYBE`, `MEAN`, `MEDICINE`, `MINUTE`, `MONDAY`, `MONEY`, `MONKEY`, `MONTH`, `MORNING`, `MOTHER`,
`MOUSE`, `NARROW`, `NEIGHBOUR`, `NEW`, `NEWSPAPER`, `NICE`, `NIGHT`, `OFFICE`, `OLD`, `ONION`,
`ORANGE`, `PAGE`, `PAINT`, `PANT`, `PAPER`, `PARENT`, `PARK`, `PATIENT`, `PEACE`, `PEACOCK`,
`PEN`, `PENCIL`, `PHOTOGRAPH`, `PIGEON`, `PINK`, `PLANE`, `PLAYER`, `PLEASED`, `POCKET`,
`POLICE`, `POOR`, `POUR`, `PRESIDENT`, `PRICE`, `PRIEST`, `QUEEN`, `QUIET`, `RACE_ETHNICITY`,
`RADIO`, `RADISH`, `RED`, `RELIGION`, `REPORTER`, `RESTAURANT`, `RICH`, `RING`, `SAD`,
`SATURDAY`, `SCHOOL`, `SCIENCE`, `SCREEN`, `SEASON`, `SECOND`, `SECRETARY`, `SHALLOW`, `SHE`,
`SHIRT`, `SHOES`, `SHORT`, `SICK`, `SIGN`, `SISTER`, `SKIRT`, `SLOW`, `SMALL_LITTLE`, `SOAP`,
`SOFT`, `SOLDIER`, `SON`, `SPARROW`, `SPORT`, `SPRING`, `STILL`, `STORE_OR_SHOP`,
`STREET_OR_ROAD`, `STRONG`, `STUDENT`, `SUIT`, `SUMMER`, `SUNDAY`, `SWITCH`, `TABLE`, `TALL`,
`TEA`, `TEACHER`, `TEAM`, `TECHNOLOGY`, `TELEPHONE`, `TELEVISION`, `TEMPLE`, `THANK_YOU`, `THEY`,
`THICK`, `THIN`, `THURSDAY`, `TIGER`, `TIGHT`, `TIME`, `TODAY`, `TOMORROW`, `TOOL`, `TRAIN`,
`TRAIN_STATION`, `TRAIN_TICKET`, `TRANSPORTATION`, `TRUCK`, `TUESDAY`, `TURTLE`, `T_SHIRT`,
`UGLY`, `UMBRELLA`, `UNCLE`, `UNIVERSITY`, `VEGETABLES`, `VOLCANO`, `WAITER`, `WAR`, `WARM`,
`WE`, `WEAK`, `WEDNESDAY`, `WEEK`, `WET`, `WHAT_IS_YOUR_NAME`, `WHITE`, `WIDE`, `WIFE`, `WINDOW`,
`WINTER`, `WOMAN`, `WRITER`, `WRONG`, `YEAR`, `YELLOW`, `YESTERDAY`, `YOU`, `YOUNG`,
`YOU_PLURAL`

## Expanded introduction static model - 32 distinct labels

Artifact: [`intro_static_pose_medoids.npz`](../app/backend/models/intro_static_pose_medoids.npz) and its
[model card](../app/backend/models/intro_static_pose_model_card.json). Only labels marked **live** pass its
current live-threshold gate; the rest are **preview only** and will not be returned as live predictions.
This experimental model does not recognize motion, sentences, or alphabet spelling.

`ANGRY` (preview only), `COLLEGE_SCHOOL` (preview only), `CRYING` (preview only), `DO` (live), `FOOD` (live), `FRIEND` (preview only), `FROM` (live), `GOOD` (preview only), `HELLO_HI` (preview only), `HELP` (live), `HOW` (preview only), `HURT` (preview only), `I_ME_MINE_MY` (live), `MEET` (preview only), `NAME` (live), `NICE` (live), `NOT` (preview only), `OLD_AGE` (preview only), `PLEASE` (preview only), `REALLY` (preview only), `SOME ONE` (preview only), `SORRY` (preview only), `TELL` (preview only), `THANK` (live), `THAT` (preview only), `TODAY` (preview only), `UNDERSTAND` (preview only), `WANT` (preview only), `WATER` (preview only), `WELCOME` (preview only), `WHAT` (live), `YOU` (live)

## Fingerspelling alphabet - 26 labels

Artifact: [`isl_letters_v1/model_card.json`](../app/backend/models/isl_letters_v1/model_card.json).

`A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`,
`T`, `U`, `V`, `W`, `X`, `Y`, `Z`

## Phrase model - 99 labels

Artifact: [`isl_phrases_v1_bilstm/model_card.json`](../app/backend/models/isl_phrases_v1_bilstm/model_card.json).
Phrase recognition returns candidate guesses for the user to select; it does not automatically commit a phrase.

`ARE_YOU_FREE_TODAY`, `ARE_YOU_HIDING_SOMETHING`, `BRING_WATER_FOR_ME`, `CAN_I_HELP_YOU`,
`CAN_YOU_REPEAT_THAT_PLEASE`, `COMB_YOUR_HAIR`, `CONGRATULATIONS`,
`COULD_YOU_PLEASE_TALK_SLOWER`, `DO_ME_A_FAVOUR`, `DO_NOT_ABUSE_HIM`, `DO_NOT_BE_STUBBORN`,
`DO_NOT_HURT_ME`, `DO_NOT_MAKE_ME_ANGRY`, `DO_NOT_TAKE_IT_TO_THE_HEART`, `DO_NOT_WORRY`,
`DO_YOU_NEED_SOMETHING`, `GO_AND_SLEEP`, `HAD_YOUR_FOOD`, `HELP_ME`, `HE_CAME_BY_TRAIN`,
`HE_IS_GOING_INTO_THE_ROOM`, `HE_IS_ON_THE_WAY`, `HE_SHE_IS_MY_FRIEND`,
`HE_WOULD_BE_COMING_TODAY`, `HI_HOW_ARE_YOU`, `HOW_ARE_THINGS`, `HOW_CAN_I_HELP_YOU`,
`HOW_CAN_I_TRUST_YOU`, `HOW_DARE_YOU`, `HOW_OLD_ARE_YOU`,
`IT_DOES_NOT_MAKE_ANY_DIFFERENCE_TO_ME`, `IT_WAS_NICE_CHATTING_WITH_YOU`, `I_AM_AFRAID_OF_THAT`,
`I_AM_AGE`, `I_AM_CRYING`, `I_AM_FEELING_BORED`, `I_AM_FEELING_COLD`, `I_AM_FINE_THANK_YOU_SIR`,
`I_AM_HUNGRY`, `I_AM_IN_DILEMMA_WHAT_TO_DO`, `I_AM_NOT_REALLY_SURE`, `I_AM_REALLY_GRATEFUL`,
`I_AM_SITTING_IN_THE_CLASS`, `I_AM_SO_SORRY_TO_HEAR_THAT`, `I_AM_SUFFERING_FROM_FEVER`,
`I_AM_TIRED`, `I_AM_VERY_HAPPY`, `I_CAN_NOT_HELP_YOU_THERE`, `I_DO_NOT_AGREE`,
`I_DO_NOT_LIKE_IT`, `I_DO_NOT_MEAN_IT`, `I_ENJOYED_A_LOT`, `I_GOT_HURT`, `I_LIKE_YOU_I_LOVE_YOU`,
`I_NEED_WATER`, `I_PROMISE`, `I_REALLY_APPRECIATE_IT`, `I_SOMEHOW_GOT_TO_KNOW_ABOUT_IT`,
`I_WAS_STOPPED_BY_SOME_ONE`, `LET_HIM_TAKE_TIME`, `MY_NAME_IS_XXXXXXXX`, `NICE_TO_MEET_YOU`,
`NOW_ONWARDS_HE_WILL_NEVER_HURT_YOU`, `POUR_SOME_MORE_WATER_INTO_THE_GLASS`, `PREPARE_THE_BED`,
`SERVE_THE_FOOD`, `SHALL_WE_GO_OUTSIDE`, `SPEAK_SOFTLY`, `TAKE_CARE_OF_YOURSELF`,
`TELL_ME_TRUTH`, `THANK_YOU_SO_MUCH`, `THAT_IS_SO_KIND_OF_YOU`, `THIS_PLACE_IS_BEAUTIFUL`,
`TRY_TO_UNDERSTAND`, `TURN_ON_LIGHT_TURN_OFF_LIGHT`, `WEAR_THE_SHIRT`, `WE_ARE_ALL_WITH_YOU`,
`WHAT_ARE_YOU_DOING`, `WHAT_DID_YOU_TELL_HIM`, `WHAT_DO_YOU_DO`, `WHAT_DO_YOU_THINK`,
`WHAT_DO_YOU_WANT_TO_BECOME`, `WHAT_HAPPENED`, `WHAT_HAVE_YOU_PLANNED_FOR_YOUR_CAREER`,
`WHAT_IS_YOUR_PHONE_NUMBER`, `WHAT_YOU_WANT`, `WHEN_WILL_THE_TRAIN_LEAVE`, `WHERE_ARE_YOU_FROM`,
`WHICH_COLLEGESCHOOL_ARE_YOU_FROM`, `WHO_ARE_YOU`, `WHY_ARE_YOU_ANGRY`, `WHY_ARE_YOU_CRYING`,
`WHY_ARE_YOU_DISAPPOINTED`, `YOU_ARE_BAD`, `YOU_ARE_GOOD`, `YOU_ARE_WELCOME`, `YOU_CAN_DO_IT`,
`YOU_DO_ANYTHING_I_DO_NOT_CARE`, `YOU_NEED_A_MEDICINE_TAKE_THIS_ONE`

## Limitations

These labels come from trained model class lists. They do not prove live-webcam accuracy, independent ISL review,
or that a separate per-sign video/image sample is available in this repository. See [`SIGN_LIBRARY.md`](SIGN_LIBRARY.md)
for model evaluation and data limitations.
