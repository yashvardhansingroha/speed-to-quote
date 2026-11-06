"""Word lists for the SYNTHETIC data generator. Everything here is fictional.

Business names are made up by combining generic words; they are not meant to match any real
company. Phone numbers use the reserved fictional 555-01xx range; emails use example.com.
"""

FIRST_NAMES = [
    "Mike", "Dave", "Sarah", "Luis", "Priya", "Tom", "Angela", "Marcus", "Jen", "Carlos",
    "Bobby", "Tasha", "Greg", "Rosa", "Dwayne", "Kim", "Hector", "Linda", "Omar", "Beth",
    "Jamal", "Nina", "Walt", "Estelle", "Raj", "Colleen", "Terrence", "Yuki", "Frank", "Daniela",
    "Pete", "Monica", "Andre", "Gail", "Victor", "Shawna", "Lonnie", "Imani", "Roy", "Teresa",
]
LAST_NAMES = [
    "Alvarez", "Brooks", "Chen", "Dalton", "Ellis", "Foster", "Garza", "Hughes", "Ibarra",
    "Jensen", "Kowalski", "Lopez", "Mercer", "Nguyen", "Okafor", "Patel", "Quinn", "Reyes",
    "Sutton", "Tran", "Underwood", "Vega", "Walsh", "Xiong", "Young", "Zimmerman", "Boyd",
    "Cole", "Duarte", "Fischer", "Grant", "Hale", "Iverson", "Jordan", "Keller", "Lang",
]

BIZ_PREFIX = [
    "Lone Star", "Blue Ridge", "Prairie Wind", "Redline", "Copper Creek", "Ironwood", "Silver Mesa",
    "Heartland", "Summit", "Cedar Fork", "Bayou", "Granite", "Northstar", "Tallgrass", "Harbor",
    "Mesquite", "Rio Verde", "High Plains", "Two Rivers", "Big Sky", "Switchback", "Pinecrest",
    "Black Mesa", "Riverbend", "Sagebrush", "Thunder Road", "Golden Spur", "Windmill", "Oakridge",
    "Coyote Ridge", "Longhorn", "Stonebridge", "Dusty Boots", "Maple Hollow", "Eagle Pass",
    "Cottonwood", "Bluebonnet", "Timberline", "Ozark", "Delta",
]
BIZ_SUFFIX = [
    "Trucking", "Transport", "Freight", "Hauling", "Logistics", "Carriers", "Express", "Cartage",
    "Transportation", "Lines",
]
BIZ_ENTITY = ["LLC", "LLC", "LLC", "Inc", "Inc", ""]

# Fictional insurers (any resemblance to a real company is coincidental).
CARRIERS = [
    "Granite Peak Mutual", "Redwood Casualty", "Beacon Harbor Insurance", "Summit Ridge Underwriters",
    "Ironbridge Mutual", "Cobalt Plains Insurance", "Northfield Casualty", "Evergreen Indemnity",
    "Westgate Surety", "Pinnacle Row Insurance", "Lakeshore Assurance", "Copperline Mutual",
]

STATE_GROUPS = [
    ["Texas", "Oklahoma", "Arkansas", "Louisiana", "New Mexico"],
    ["Ohio", "Indiana", "Michigan", "Kentucky", "Pennsylvania", "Illinois"],
    ["Georgia", "Florida", "Alabama", "Tennessee", "South Carolina", "North Carolina"],
    ["California", "Nevada", "Arizona", "Oregon", "Utah"],
    ["Nebraska", "Kansas", "Missouri", "Iowa", "Colorado", "South Dakota"],
    ["Minnesota", "Wisconsin", "North Dakota", "Montana", "Illinois"],
    ["New York", "New Jersey", "Pennsylvania", "Virginia", "Maryland"],
]
ABBR = {
    "Texas": "TX", "Oklahoma": "OK", "Arkansas": "AR", "Louisiana": "LA", "New Mexico": "NM",
    "Ohio": "OH", "Indiana": "IN", "Michigan": "MI", "Kentucky": "KY", "Pennsylvania": "PA",
    "Illinois": "IL", "Georgia": "GA", "Florida": "FL", "Alabama": "AL", "Tennessee": "TN",
    "South Carolina": "SC", "North Carolina": "NC", "California": "CA", "Nevada": "NV",
    "Arizona": "AZ", "Oregon": "OR", "Utah": "UT", "Nebraska": "NE", "Kansas": "KS",
    "Missouri": "MO", "Iowa": "IA", "Colorado": "CO", "South Dakota": "SD", "Minnesota": "MN",
    "Wisconsin": "WI", "North Dakota": "ND", "Montana": "MT", "New York": "NY",
    "New Jersey": "NJ", "Virginia": "VA", "Maryland": "MD",
}

# (descriptor used in "we run ...", commodity phrase, vehicle types)
OPERATIONS = [
    ("a dry van freight operation", "general freight", ["dry van tractors", "tractors"]),
    ("a refrigerated hauling company", "produce", ["reefer tractors", "reefers"]),
    ("a flatbed outfit", "steel and building materials", ["flatbed trucks", "flatbeds"]),
    ("a small dump truck business", "gravel and aggregates", ["dump trucks"]),
    ("a last-mile delivery business", "packages", ["box trucks", "cargo vans"]),
    ("an auto hauling company", "vehicles", ["car haulers", "hotshots"]),
    ("a livestock hauling operation", "cattle", ["pot trailers", "tractors"]),
    ("a fuel delivery company", "gasoline and diesel", ["tankers"]),
    ("a hotshot trucking business", "machinery", ["pickups", "hotshots"]),
    ("a construction debris hauling company", "debris", ["roll-off trucks", "dump trucks"]),
    ("a household goods moving company", "household goods", ["box trucks", "straight trucks"]),
    ("a grain hauling operation", "grain", ["hopper trailers", "tractors"]),
]
MAKES = ["Freightliner", "Peterbilt", "Kenworth", "Volvo", "International", "Mack", "Ford", "Isuzu", "Hino"]
MODELS = {"Freightliner": "Cascadia", "Peterbilt": "579", "Kenworth": "T680", "Volvo": "VNL",
          "International": "LT", "Mack": "Anthem", "Ford": "F-550", "Isuzu": "NPR", "Hino": "268"}

COVERAGE_PHRASES = [
    "primary liability and physical damage", "liability, cargo and physical damage",
    "auto liability and motor truck cargo", "liability and cargo", "primary liability",
    "physical damage and non-trucking liability", "liability, physical damage, and trailer interchange",
    "cargo and general liability",
]
VAGUE_COVERAGE = ["full coverage", "the usual package", "whatever is required", "state minimum"]

# Very short / ambiguous messages. {first} and {phone} are filled in.
SHORT_MESSAGES = [
    "Hi", "need a quote", "insurance?", "Trucking insurance pls", "Call me back {phone}",
    "Quote for 1 truck asap", "looking for coverage", "can someone call me", "Need COI",
    "Hello?? anyone there", "Please call, it's {first}", "interested in your services",
    "How much for a truck policy", "need to talk about my policy", "question about coverage",
    "renewal coming up, call me", "got your number from a friend",
]
AMBIGUOUS_MESSAGES = [
    "Hi, this is {first}. We spoke a while back about the thing. Can you call me when you get a chance?",
    "Hello, I'm reaching out regarding my account. Please give me a call at {phone}.",
    "Do you guys handle this kind of stuff? Not sure who else to ask.",
    "Got your info from another driver. What do you offer and what do you need from me?",
    "I think something changed with our situation and I need to talk to someone.",
    "Can you let me know what the next step is? Thanks, {first}.",
    "Hey, {first} here. Following the call yesterday I'm not sure what I should send over.",
    "Need help with some paperwork, call me.",
]

# Context sentences that add realism (and a little noise for the extractors).
FILLER = [
    "We have been in business since {year}.", "A friend of mine in the industry gave me your name.",
    "Our previous broker retired and we are starting fresh.", "Business has been steady this year.",
    "We just picked up a new contract with a local distributor.", "I found you online.",
    "I'm out on the road most of the day so email is best.", "We are hoping to add another truck next spring.",
    "Our accountant told us to look at our insurance costs.", "Sorry for the long message.",
    "Happy to jump on a call anytime after 3.", "We have been with the same company for a long time.",
    "This is the first time I'm doing this on my own.", "The shipper we work with requires proof of coverage.",
    "Fuel costs are killing us so I'm trying to cut where I can.", "We are a family operation.",
]
