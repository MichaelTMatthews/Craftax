# To-Do
- [x] Direction towards teammates
  - [x] Symbolic
  - [x] Pixels
- [x] Different Color Players
  - [x] Symbolic
  - [x] Pixels
- [x] Different Color Chests
  - [x] Game Logic
  - [x] Symbolic
  - [x] Pixels
- [] Dashboard
  - [] Pixels
    - [x] Player Textures
    - [x] Health
    - [x] Direction
    - [] Message
- [] Messaging
- [x] Trading

# Legacy
- (Fixed) Passive mobs don't moves
- (Fixed) Mobs when close to player move over player
- (Fixed) Skeleton only shoots if player is below or above them, not side by side.
- (NOT A BUG) When player dies, mobs around player disappear
- (FIXED) Player health regenerates when health
  - Disable update player intrinsics if player not alive
  - Possibly reset all player intrinsics if player dies

Nice to adds:
- Multiple chests (one chest per player)
- See what parameters need to be reset when reviving player
- Add achievement for reviving player?
- Not despawn immediately after killing player


Steps to success:
- (DONE) Evaluate Do Action
- (DONE) Disable update player intriniscs if player not alive
- Check how we interact with ladder
  - Can you place block on ladder?
  - Can players walk on ladder?
  - Do players spawn on ladders?
- (DONE) Add revive
- Profile all parts (reset, step, render), with different player count, and compare with original
