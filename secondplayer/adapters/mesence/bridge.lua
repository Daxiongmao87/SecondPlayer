-- SecondPlayer MesenCE bridge.
--
-- This file intentionally contains no AI logic. It is an adapter shim between
-- stock MesenCE's Lua API and the external SecondPlayer daemon.

local socket = require("socket.core")

local host = os.getenv("SECONDPLAYER_BRIDGE_HOST") or "127.0.0.1"
local port = tonumber(os.getenv("SECONDPLAYER_BRIDGE_PORT") or "0")
if port == 0 then
  error("SECONDPLAYER_BRIDGE_PORT is required")
end

local client = assert(socket.tcp())
client:settimeout(10)
assert(client:connect(host, port))
client:settimeout(0)

local rx = ""
local pending_captures = {}
local pending_waits = {}
local pending_states = {}
local ready_states = {}
local pending_load = nil
local pending_load_apply = nil
local loads_done = {}
local state_cb_ref = nil
local state_cb_fired = false
local desired = {}
local setinput_mode = nil
local mode_announced = false
local closing = false
-- When pulse_frames > 0, AI ports auto-release to neutral once more than
-- that many emulated frames pass without a fresh INPUT command. This turns
-- each slow policy decision into a short tap instead of a ~1s hold.
local pulse_frames = tonumber(os.getenv("SECONDPLAYER_PULSE_FRAMES") or "0") or 0
local frame_count = 0
local last_input_frame = {}

local BUTTONS = {
  [0] = "up",
  [1] = "down",
  [2] = "left",
  [3] = "right",
  [4] = "a",
  [5] = "b",
  [6] = "x",
  [7] = "y",
  [8] = "l",
  [9] = "r",
  [10] = "start",
  [11] = "select",
}

local function send_all(data)
  local first = 1
  while first <= #data do
    local sent, err, last = client:send(data, first)
    if sent then
      first = sent + 1
    elseif last and last >= first then
      first = last + 1
    elseif err == "timeout" then
      -- Localhost backpressure should be brief. Yield to the next retry.
    else
      error("SecondPlayer bridge send failed: " .. tostring(err))
    end
  end
end

local function send_line(line)
  send_all(line .. "\n")
end

local function mask_to_input(mask)
  local input = {}
  for bit = 0, 11 do
    local name = BUTTONS[bit]
    input[name] = (mask & (1 << bit)) ~= 0
  end
  return input
end

local function input_to_mask(input)
  local mask = 0
  for bit = 0, 11 do
    local name = BUTTONS[bit]
    if input[name] then
      mask = mask | (1 << bit)
    end
  end
  return mask
end

-- MesenCE 2.2.1/current source has an extra lua_settop() in LuaApi::SetInput
-- which shifts the port/subport reads. We detect the calling convention at
-- runtime and use the appropriate stock API invocation without patching Mesen.
local function detect_setinput_mode()
  if setinput_mode ~= nil then
    return
  end

  local p0 = emu.getInput(0)
  local p1 = emu.getInput(1)
  local original0 = p0.select == true
  local original1 = p1.select == true
  local probe = not original1

  emu.setInput({ select = probe }, 1)
  local after0 = emu.getInput(0).select == true
  local after1 = emu.getInput(1).select == true

  if after1 == probe then
    setinput_mode = "fixed"
    emu.setInput({ select = original1 }, 1)
    emu.setInput({ select = original0 }, 0)
  else
    setinput_mode = "legacy-2.2.1"
    -- With the affected API, the third Lua argument is read as the port.
    emu.setInput({ select = original1 }, 0, 1)
    emu.setInput({ select = original0 }, 0)
  end
end

local function set_port(port_number, buttons)
  detect_setinput_mode()
  if setinput_mode == "fixed" then
    emu.setInput(buttons, port_number)
  elseif port_number == 0 then
    emu.setInput(buttons, 0)
  else
    emu.setInput(buttons, 0, port_number)
  end
end

-- MesenCE only permits createSavestate/loadSavestate inside an "exec" memory
-- callback. Register a one-shot full-range callback; it fires at the next
-- executed CPU instruction, stashes the result, and is removed from the next
-- endFrame so normal emulation pays no per-instruction overhead.
local function ensure_state_callback()
  if state_cb_ref ~= nil then return end
  state_cb_ref = emu.addMemoryCallback(function()
    for _, id in ipairs(pending_states) do
      ready_states[#ready_states + 1] = { id = id, data = emu.createSavestate() }
    end
    pending_states = {}
    if pending_load_apply ~= nil then
      emu.loadSavestate(pending_load_apply.data)
      loads_done[#loads_done + 1] = pending_load_apply.id
      pending_load_apply = nil
    end
    state_cb_fired = true
  end, emu.callbackType.exec, 0, 0xFFFFFF)
end

local function handle_line(line)
  local command, rest = line:match("^(%S+)%s*(.*)$")
  if command == "PING" then
    local id = rest:match("^(%d+)$")
    if id then send_line("PONG " .. id) end
    return
  end

  if command == "CAPTURE" then
    local id = tonumber(rest)
    if id then pending_captures[#pending_captures + 1] = id end
    return
  end

  if command == "WAITFRAMES" then
    local id, count = rest:match("^(%d+)%s+(%d+)$")
    id = tonumber(id)
    count = tonumber(count)
    if id and count and count >= 1 then
      pending_waits[#pending_waits + 1] = { id = id, remaining = count }
    end
    return
  end

  if command == "INPUT" then
    local port_number, mask = rest:match("^(%d+)%s+(%d+)$")
    port_number = tonumber(port_number)
    mask = tonumber(mask)
    if port_number and mask and port_number >= 0 and port_number <= 1 then
      desired[port_number] = mask
      last_input_frame[port_number] = frame_count
    end
    return
  end

  if command == "GETINPUT" then
    local id, port_number = rest:match("^(%d+)%s+(%d+)$")
    id = tonumber(id)
    port_number = tonumber(port_number)
    if id and port_number and port_number >= 0 and port_number <= 1 then
      local mask = input_to_mask(emu.getInput(port_number))
      send_line("INPUTSTATE " .. tostring(id) .. " " .. tostring(mask))
    end
    return
  end

  if command == "SAVESTATE" then
    local id = tonumber(rest)
    if id then
      pending_states[#pending_states + 1] = id
      ensure_state_callback()
    end
    return
  end

  if command == "LOADSTATE" then
    local id, length = rest:match("^(%d+)%s+(%d+)$")
    id = tonumber(id)
    length = tonumber(length)
    if id and length and length >= 1 and length <= 67108864 then
      -- The daemon sends no further commands until OK, so any bytes already
      -- buffered after this header line belong to the payload.
      pending_load = { id = id, remaining = length, chunks = {} }
    else
      send_line("ERROR invalid LOADSTATE header")
    end
    return
  end

  if command == "SHUTDOWN" then
    local id = tonumber(rest)
    if id then send_line("OK " .. tostring(id)) end
    closing = true
    emu.stop(0)
    return
  end

  send_line("ERROR unknown command: " .. tostring(command))
end

-- Move buffered bytes into the in-progress LOADSTATE payload. Payload bytes
-- are binary (they may contain newlines), so while a load is in progress the
-- line splitter below stays off.
local function consume_load_payload()
  if pending_load == nil then return end
  if #rx > 0 then
    local take = math.min(#rx, pending_load.remaining)
    pending_load.chunks[#pending_load.chunks + 1] = rx:sub(1, take)
    rx = rx:sub(take + 1)
    pending_load.remaining = pending_load.remaining - take
  end
  if pending_load.remaining == 0 then
    local id = pending_load.id
    local data = table.concat(pending_load.chunks)
    pending_load = nil
    pending_load_apply = { id = id, data = data }
    ensure_state_callback()
  end
end

local function pump()
  while true do
    consume_load_payload()
    local chunk, err, partial = client:receive(4096)
    local data = chunk or partial
    if data and #data > 0 then
      if pending_load ~= nil then
        -- Route payload bytes straight into chunks; growing rx 4KB at a
        -- time over a multi-MB upload would be quadratic.
        local take = math.min(#data, pending_load.remaining)
        pending_load.chunks[#pending_load.chunks + 1] = data:sub(1, take)
        pending_load.remaining = pending_load.remaining - take
        if take < #data then rx = rx .. data:sub(take + 1) end
        consume_load_payload()
      else
        rx = rx .. data
        while true do
          local nl = rx:find("\n", 1, true)
          if not nl then break end
          local line = rx:sub(1, nl - 1)
          rx = rx:sub(nl + 1)
          if #line > 0 then handle_line(line) end
          if pending_load ~= nil then break end
        end
      end
    end
    if err == "closed" then
      closing = true
      emu.stop(0)
      return
    end
    if err == "timeout" or chunk == nil then
      return
    end
  end
end

local function send_pending_captures()
  if #pending_captures == 0 then return end
  -- The first endFrame after load can yield an empty screenshot before the
  -- renderer has presented anything. Hold pending ids until a real PNG is
  -- available; the daemon-side capture timeout still bounds the wait.
  local png = emu.takeScreenshot()
  if type(png) ~= "string" or #png == 0 then return end
  for _, id in ipairs(pending_captures) do
    send_line("FRAME " .. tostring(id) .. " " .. tostring(#png))
    send_all(png)
  end
  pending_captures = {}
end

send_line("HELLO 1")

emu.addEventCallback(function()
  if closing then return end
  frame_count = frame_count + 1
  detect_setinput_mode()
  if not mode_announced then
    send_line("MODE " .. setinput_mode)
    mode_announced = true
  end
  -- Apply before pumping: Mesen clears pad bits every frame, so a GETINPUT
  -- answered before this loop would report pre-apply (neutral) state.
  for port_number, mask in pairs(desired) do
    local effective = mask
    if pulse_frames > 0 and frame_count - (last_input_frame[port_number] or 0) > pulse_frames then
      effective = 0
    end
    set_port(port_number, mask_to_input(effective))
  end
  pump()
end, emu.eventType.inputPolled)

local function send_pending_waits()
  local ready = {}
  for i = #pending_waits, 1, -1 do
    local wait = pending_waits[i]
    wait.remaining = wait.remaining - 1
    if wait.remaining <= 0 then
      ready[#ready + 1] = wait.id
      table.remove(pending_waits, i)
    end
  end
  for _, id in ipairs(ready) do
    send_line("WAITED " .. tostring(id))
  end
end

local function send_pending_states()
  if state_cb_fired and state_cb_ref ~= nil then
    emu.removeMemoryCallback(state_cb_ref, emu.callbackType.exec, 0, 0xFFFFFF)
    state_cb_ref = nil
    state_cb_fired = false
  end
  for _, saved in ipairs(ready_states) do
    send_line("STATE " .. tostring(saved.id) .. " " .. tostring(#saved.data))
    send_all(saved.data)
  end
  ready_states = {}
  for _, id in ipairs(loads_done) do
    send_line("OK " .. tostring(id))
  end
  loads_done = {}
end

emu.addEventCallback(function()
  if closing then return end
  pump()
  send_pending_waits()
  send_pending_states()
  send_pending_captures()
end, emu.eventType.endFrame)
