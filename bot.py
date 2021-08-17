
from twitchio.ext import commands
import sekrets
import asyncio
from datetime import date
import webcrawler2wordoftheday
import lang_model


class Bot(commands.Bot):

    def __init__(self):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        # prefix can be a callable, which returns a list of strings or a string...
        # initial_channels can also be a callable which returns a list of strings...
        super().__init__(
            token=sekrets.secrets['TMI_TOKEN'],
            client_id=sekrets.secrets['CLIENT_ID'],
            nick=sekrets.secrets['BOT_NICK'],
            prefix=sekrets.secrets['BOT_PREFIX'],
            initial_channels=[sekrets.secrets['CHANNEL']]
        )

    async def event_ready(self):
        # Notify us when everything is ready!
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Here we have a command hello, we can invoke our command with our prefix and command name
        # e.g ?hello
        # We can also give our commands aliases (different names) to invoke with.

        # Send a hello back!
        # Sending a reply back to the channel is easy... Below is an example.
        await ctx.send(f'Hello {ctx.author.name}!!')

    @commands.command(name="hail", aliases=["Hail", "HAIL"])
    async def hail(self, ctx: commands.Context):
        # this is a simple patois greeting to whoever uses the command !patois.
        await ctx.send(f'Wah gwan Genna finite6Ollie {ctx.author.name}  ')

    @commands.command()
    async def today(self, ctx: commands.Context):
        # this will read from a txt file containing todays desired order of events
        with open("todays.txt", "r") as f:
            data = f.read()
        today = date.today()
        # print("Today's date:", today.strftime("%Y-%m-%d))
        await ctx.send(f' Today\'s date:{today.strftime("%Y-%m-%d")} - {data}')
        # print(f.read())

    @commands.command()
    async def discord(self, ctx: commands.Context):
        # generates the discord link in the chat
        disc_link = 'discord.gg/fpTXj92M8m'
        await ctx.send(f'1st rule of the discord is we dont talk about the discord {disc_link} ')

    @commands.command()
    async def translate(self, ctx: commands.Context):
        # translates simple phrases
        _, query = ctx.message.content.split(" ", maxsplit=1)
        message_sent = lang_model.translate(query)

        await ctx.send(f'{message_sent}')

    @commands.command()
    async def patois(self, ctx: commands.Context):
        # this comand will provide a the Patois word of the day
        # and its respective english definition
        with open("today.txt", "r") as f:
            first_line = f.readline()
            # print(first_line)

        if(first_line != date.today().strftime("%Y-%m-%d")):
            await ctx.send(f' Learn dis : {webcrawler2wordoftheday.wrd_of_day}')
            await asyncio.sleep(3)
            await ctx.send(f' It means : {webcrawler2wordoftheday.wrd_defn}')
            with open("today.txt", "w") as f:
                f.write(date.today().strftime("%Y-%m-%d") + "\n")
                f.write(webcrawler2wordoftheday.wrd_of_day + "\n")
                f.write(webcrawler2wordoftheday.wrd_defn + "\n")
        else:

            with open("today.txt", "r") as f:
                first_line = f.readline()
                second_line = f.readline()
                third_line = f.readline()

                await ctx.send(f' Learn dis : {second_line}')
                await asyncio.sleep(3)
                await ctx.send(f' It means : {third_line}')


bot = Bot()
bot.run()
