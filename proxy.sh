#!/bin/bash

e=$(networksetup -getsocksfirewallproxy wi-fi | grep "No")

if [ -n "$e" ]; then
  echo "Turning on proxy"
  # sudo networksetup -setstreamingproxystate wi-fi on
  sudo networksetup -setsocksfirewallproxystate wi-fi on
  echo 'display notification "ON" with title "proxy.sh"'  | osascript
  # sudo networksetup -setwebproxystate wi-fi on
else
  echo "Turning off proxy"
  # sudo networksetup -setstreamingproxystate wi-fi off
  sudo networksetup -setsocksfirewallproxystate wi-fi off
  echo 'display notification "OFF" with title "proxy.sh"' | osascript

  # sudo networksetup -setwebproxystate wi-fi off
fi