"""DATORYX Plugin Manager - Dynamic plugin loading and lifecycle."""
import os
import sys
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import asyncio


@dataclass
class Plugin:
    name: str
    version: str
    description: str
    author: str
    module: Any
    hooks: Dict[str, List[Callable]]
    config: Dict[str, Any]
    enabled: bool = True
    loaded_at: Optional[str] = None


class PluginManager:
    """Load, manage, and execute plugins with hook system."""

    def __init__(self, plugin_dir: str = "./plugins"):
        self._plugins: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._plugin_dir = Path(plugin_dir)
        self._hook_results: Dict[str, List[Any]] = {}

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin directory."""
        if not self._plugin_dir.exists():
            return []

        plugins = []
        for item in self._plugin_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                plugins.append(item.name)
            elif item.suffix == ".py" and item.name != "__init__.py":
                plugins.append(item.stem)
        return plugins

    def load_plugin(self, name: str, config: Dict[str, Any] = None) -> bool:
        """Load a plugin by name."""
        try:
            plugin_path = self._plugin_dir / name

            if plugin_path.is_dir():
                spec = importlib.util.spec_from_file_location(
                    name, plugin_path / "__init__.py"
                )
            else:
                spec = importlib.util.spec_from_file_location(
                    name, self._plugin_dir / f"{name}.py"
                )

            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)

            # Extract plugin metadata
            plugin = Plugin(
                name=getattr(module, "PLUGIN_NAME", name),
                version=getattr(module, "PLUGIN_VERSION", "0.1.0"),
                description=getattr(module, "PLUGIN_DESCRIPTION", ""),
                author=getattr(module, "PLUGIN_AUTHOR", "Unknown"),
                module=module,
                hooks=getattr(module, "HOOKS", {}),
                config=config or {}
            )

            # Register hooks
            for hook_name, handlers in plugin.hooks.items():
                if hook_name not in self._hooks:
                    self._hooks[hook_name] = []
                self._hooks[hook_name].extend(handlers)

            # Initialize if init function exists
            if hasattr(module, "initialize"):
                init_func = getattr(module, "initialize")
                if asyncio.iscoroutinefunction(init_func):
                    asyncio.create_task(init_func(plugin.config))
                else:
                    init_func(plugin.config)

            self._plugins[name] = plugin
            return True

        except Exception as e:
            print(f"[PluginManager] Failed to load plugin {name}: {e}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.pop(name, None)
        if not plugin:
            return False

        # Unregister hooks
        for hook_name, handlers in plugin.hooks.items():
            for handler in handlers:
                if handler in self._hooks.get(hook_name, []):
                    self._hooks[hook_name].remove(handler)

        # Cleanup if function exists
        if hasattr(plugin.module, "cleanup"):
            try:
                cleanup = getattr(plugin.module, "cleanup")
                if asyncio.iscoroutinefunction(cleanup):
                    asyncio.create_task(cleanup())
                else:
                    cleanup()
            except Exception:
                pass

        if name in sys.modules:
            del sys.modules[name]

        return True

    async def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all handlers for a hook."""
        results = []
        handlers = self._hooks.get(hook_name, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"[PluginManager] Hook {hook_name} error: {e}")

        self._hook_results[hook_name] = results
        return results

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins."""
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "enabled": p.enabled
            }
            for p in self._plugins.values()
        ]

    def get_hook_results(self, hook_name: str) -> List[Any]:
        """Get results from last hook execution."""
        return self._hook_results.get(hook_name, [])
