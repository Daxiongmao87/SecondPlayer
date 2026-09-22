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
local desired = {}
local setinput_mode = nil
local mode_announced = false
local closing = false

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
  send_all(line .. "
")
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

  if command == "INPUT" then
    local port_number, mask = rest:match("^(%d+)%s+(%d+)$")
    port_number = tonumber(port_number)
    mask = tonumber(mask)
    if port_number and mask and port_number >= 0 and port_number <= 1 then
      desired[port_number] = mask
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

  if command == "SHUTDOWN" then
    local id = tonumber(rest)
    if id then send_line("OK " .. tostring(id)) end
    closing = true
    emu.stop(0)
    return
  end

  send_line("ERROR unknown command: " .. tostring(command))
end

local function pump()
  while true do
    local chunk, err, partial = client:receive(4096)
    local data = chunk or partial
    if data and #data > 0 then
      rx = rx .. data
      while true do
        local nl = rx:find("
", 1, true)
        if not nl then break end
        local line = rx:sub(1, nl - 1)
        rx = rx:sub(nl + 1)
        if #line > 0 then handle_line(line) end
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
  local png = emu.takeScreenshot()
  for _, id in ipairs(pending_captures) do
    send_line("FRAME " .. tostring(id) .. " " .. tostring(#png))
    send_all(png)
  end
  pending_captures = {}
end

send_line("HELLO 1")

emu.addEventCallback(function()
  if closing then return end
  pump()
  detect_setinput_mode()
  if not mode_announced then
    send_line("MODE " .. setinput_mode)
    mode_announced = true
  end
  for port_number, mask in pairs(desired) do
    set_port(port_number, mask_to_input(mask))
  end
end, emu.eventType.inputPolled)

emu.addEventCallback(function()
  if closing then return end
  pump()
  send_pending_captures()
end, emu.eventType.endFrame)
