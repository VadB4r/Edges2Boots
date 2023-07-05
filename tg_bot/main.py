from bot import bot
import asyncio


def main():
    asyncio.run(bot.polling())


if __name__ == '__main__':
    main()
