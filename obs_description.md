## Craftax Symbolic Observation Description

The craftax symbolic observation is a flat array of shape (8268,)

### [0:8217] - Flattened map representation
The original map representation can be recovered by reshaping to (9, 11, 83)
This represents the 9x11 subset of the map visible to the agent.

Each tile is split as follows:

[0:37] - 1-hot vector for Block ID

[37:42] - 1-hot vector for Item ID

[42:82] - 1-hot vector for Mob ID.  There are 5 types of mobs (ordered as melee, passive, ranged, enemy_projectile, friendly_projectile).
Each of these has 8 types.

[83] - light level (this corresponds to the light level on dark levels.  This is always 1 on light levels.  On the overworld this value will still be 1 even at night).


### [8217:8233] - Inventory

[8217] - sqrt(wood) / 10.0

[8218] - sqrt(stone) / 10.0

[8219] - sqrt(coal) / 10.0

[8220] - sqrt(iron) / 10.0

[8221] - sqrt(diamond) / 10.0

[8222] - sqrt(sapphire) / 10.0

[8223] - sqrt(ruby) / 10.0

[8224] - sqrt(sapling) / 10.0

[8225] - sqrt(torches) / 10.0

[8226] - sqrt(arrows) / 10.0

[8227] - books / 2.0

[8228] - pickaxe_level / 4.0

[8229] - sword_level / 4.0

[8230] - sword_enchantment (0=none, 1=fire, 2=ice)

[8231] - bow_enchantment (0=none, 1=fire, 2=ice)

[8232] - bow?


### [8233:8239] - Potions

[8233] - red_potion / 10.0

[8234] - green_potion / 10.0

[8235] - blue_potion / 10.0

[8236] - pink_potion / 10.0

[8237] - cyan_potion / 10.0

[8238] - yellow_potion / 10.0


### [8239:8248] - Intrinsics
[8239] - health / 10.0

[8240] - food / 10.0

[8241] - drink / 10.0

[8242] - energy / 10.0

[8243] - mana / 10.0

[8244] - xp / 10.0

[8245] - dexterity / 10.0

[8246] - strength / 10.0

[8247] - intelligence / 10.0


### [8248:8252] - Direction

One-hot representation of direction

[8248] - left?

[8249] - right?

[8250] - up?

[8251] - down?


### [8252:8260] - Armour

[8252] - Helmet level (0=none, 0.5=iron, 1=diamond)

[8253] - Chestplate level (0=none, 0.5=iron, 1=diamond)

[8254] - Leggings level (0=none, 0.5=iron, 1=diamond)

[8255] - Boots level (0=none, 0.5=iron, 1=diamond)

[8256] - Helmet enchantment (0=none, 1=fire, 2=ice)

[8257] - Chestplate enchantment (0=none, 1=fire, 2=ice)

[8258] - Leggings enchantment (0=none, 1=fire, 2=ice)

[8259] - Boots enchantment (0=none, 1=fire, 2=ice)


### [8260:8268] - Special Values

[8260] - light_level (this corresponds to day/night in overworld)

[8261] - is_sleeping?

[8262] - is_resting?

[8263] - learned_fireball?

[8264] - learned_iceball?

[8265] - current_floor / 10.0 

[8266] - is_current_floor_down_ladder_open?

[8267] - is_boss_vulnerable?
