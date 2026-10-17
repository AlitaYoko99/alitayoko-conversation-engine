# WordPress integration

The AlitaYoko Conversations AI Generator is published as a **BETA feature**.

## Plugin settings

In WordPress:

1. Open **Conversations → AI Settings**.
2. Select **AlitaYoko Conversation Engine (self-hosted)**.
3. Enter the public HTTPS Engine URL.
4. Enter the same API key configured as `AY_API_KEY` on the engine.
5. Save and generate a test conversation.

Example endpoint:

```text
https://engine.example.com
```

## Managed configuration

The plugin also supports `wp-config.php` overrides:

```php
define( 'ALYCON_GENERATOR_BACKEND', 'engine' );
define( 'ALYCON_ENGINE_URL', 'https://engine.example.com' );
define( 'ALYCON_ENGINE_API_KEY', 'replace-with-your-key' );
```

Do not publish the API key or commit it to a repository.

## Research

The plugin exposes Research only when the self-hosted AlitaYoko Engine backend is selected. Research is semantically separate from visible link attachments:

- Research supplies evidence/facts to the generator.
- Links control visible link suggestions in the generated conversation.

Research requires the engine to have a configured SearXNG endpoint.

## Plugin

AlitaYoko Conversations is available from the WordPress.org Plugin Directory:

https://wordpress.org/plugins/alitayoko-conversations/
