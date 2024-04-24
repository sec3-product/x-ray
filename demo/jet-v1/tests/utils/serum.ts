import { Market, DexInstructions } from "@project-serum/serum";
import {
    Keypair,
    LAMPORTS_PER_SOL,
    PublicKey,
    SystemProgram,
    TransactionInstruction,
} from "@solana/web3.js";
import { BN } from "@project-serum/anchor";
import { Token, TOKEN_PROGRAM_ID, AccountLayout as TokenAccountLayout } from "@solana/spl-token";
import { TestUtils, toPublicKeys } from ".";
import { TestToken } from "app/src/jet-test";

export const DEX_ID = new PublicKey("9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin");

export const DEX_ID_DEVNET = new PublicKey("DESVgJVGajEgKGXhb6XmqDHGz3VjdgP7rEVESBgxmroY");

export class SerumUtils {
    private utils: TestUtils;
    private dex_id: PublicKey;

    constructor(utils: TestUtils, devnet: boolean) {
        this.utils = utils;
        this.dex_id = devnet ? DEX_ID_DEVNET : DEX_ID;
    }

    private async createAccountIx(
        account: PublicKey,
        space: number,
        programId: PublicKey
    ): Promise<TransactionInstruction> {
        return SystemProgram.createAccount({
            newAccountPubkey: account,
            fromPubkey: this.utils.payer().publicKey,
            lamports: await this.utils
                .connection()
                .getMinimumBalanceForRentExemption(space),
            space,
            programId,
        });
    }

    private async createTokenAccountIx(
        account: PublicKey
    ): Promise<TransactionInstruction> {
        return this.createAccountIx(
            account,
            TokenAccountLayout.span,
            TOKEN_PROGRAM_ID
        );
    }

    private initTokenAccountIx(
        account: PublicKey,
        mint: PublicKey,
        owner: PublicKey
    ): TransactionInstruction {
        return Token.createInitAccountInstruction(
            TOKEN_PROGRAM_ID,
            mint,
            account,
            owner
        );
    }

    /**
     * Create a new Serum market
     * @returns
     */
    public async createMarket(info: CreateMarketInfo): Promise<Market> {
        const market = Keypair.generate();
        const requestQueue = Keypair.generate();
        const eventQueue = Keypair.generate();
        const bids = Keypair.generate();
        const asks = Keypair.generate();
        const quoteDustThreshold = new BN(100);

        const [vaultOwner, vaultOwnerBump] = await this.findVaultOwner(
            market.publicKey
        );

        const [baseVault, quoteVault] = await Promise.all([
            this.utils.createTokenAccount(
                info.baseToken,
                vaultOwner,
                new BN(0)
            ),
            this.utils.createTokenAccount(
                info.quoteToken,
                vaultOwner,
                new BN(0)
                ),
            ]);
            
        const initMarketTx = this.utils.transaction().add(
            await this.createAccountIx(
                market.publicKey,
                Market.getLayout(this.dex_id).span,
                this.dex_id
            ),
            await this.createAccountIx(
                requestQueue.publicKey,
                5132,
                this.dex_id
            ),
            await this.createAccountIx(
                eventQueue.publicKey,
                262156,
                this.dex_id
            ),
            await this.createAccountIx(bids.publicKey, 65548, this.dex_id),
            await this.createAccountIx(asks.publicKey, 65548, this.dex_id),
            DexInstructions.initializeMarket(
                toPublicKeys({
                    market,
                    requestQueue,
                    eventQueue,
                    bids,
                    asks,
                    baseVault,
                    quoteVault,
                    baseMint: info.baseToken.publicKey,
                    quoteMint: info.quoteToken.publicKey,

                    baseLotSize: new BN(info.baseLotSize),
                    quoteLotSize: new BN(info.quoteLotSize),

                    feeRateBps: info.feeRateBps,
                    vaultSignerNonce: vaultOwnerBump,

                    quoteDustThreshold,
                    programId: this.dex_id,
                })
            )
        );

        await this.utils.sendAndConfirmTransaction(initMarketTx, [
            market,
            requestQueue,
            eventQueue,
            bids,
            asks,
        ]);

        return await Market.load(
            this.utils.connection(),
            market.publicKey,
            undefined,
            this.dex_id
        );
    }

    /**
     * Create a market maker account
     * @param lamports The initial lamport funding for the market maker's wallet
     * @param tokens The list of tokens and amounts to mint for the new market maker
     * @returns Details about the market maker's accounts that were created
     */
    public async createMarketMaker(
        lamports: number,
        tokens: [Token, BN][]
    ): Promise<MarketMaker> {
        const account = await this.utils.createWallet(lamports);
        const tokenAccounts = {};
        const transactions = [];

        for (const [token, amount] of tokens) {
            const publicKey = await this.utils.createTokenAccount(
                token,
                account,
                amount
            );

            tokenAccounts[token.publicKey.toBase58()] = publicKey;
        }

        return new MarketMaker(this.utils, account, tokenAccounts);
    }

    /**
     * Create a new serum market and fill it with reasonable liquidity
     * @param baseToken The base token, as in BTC
     * @param quoteToken The quote token, as in USD
     * @param marketPrice The price that bids and asks will be created at.
     * @returns 
     */
    public async createAndMakeMarket(baseToken: TestToken, quoteToken: TestToken, marketPrice: number) {
        const market = await this.createMarket({
            baseToken,
            quoteToken,
            baseLotSize: 100000,
            quoteLotSize: 100,
            feeRateBps: 22,
        });
        const marketMaker = await this.createMarketMaker(
            1 * LAMPORTS_PER_SOL,
            [
                [baseToken, baseToken.amount(100000)],
                [quoteToken, quoteToken.amount(500000)],
            ]
        );

        const bids = MarketMaker.makeOrders([[marketPrice * 0.995, 10000]]);
        const asks = MarketMaker.makeOrders([[marketPrice * 1.005, 10000]]);

        await marketMaker.placeOrders(market, bids, asks);
        return market;
    }

    async findVaultOwner(market: PublicKey): Promise<[PublicKey, BN]> {
        const bump = new BN(0);
    
        while (bump.toNumber() < 255) {
            try {
                const vaultOwner = await PublicKey.createProgramAddress(
                    [market.toBuffer(), bump.toArrayLike(Buffer, "le", 8)],
                    this.dex_id
                );
    
                return [vaultOwner, bump];
            } catch (_e) {
                bump.iaddn(1);
            }
        }
    
        throw new Error("no seed found for vault owner");
    }
    
}

export interface CreateMarketInfo {
    baseToken: Token;
    quoteToken: Token;
    baseLotSize: number;
    quoteLotSize: number;
    feeRateBps: number;
}

export interface Order {
    price: number;
    size: number;
}

export class MarketMaker {
    public account: Keypair;
    public tokenAccounts: { [mint: string]: PublicKey };

    private utils: TestUtils;

    constructor(
        utils: TestUtils,
        account: Keypair,
        tokenAccounts: { [mint: string]: PublicKey }
    ) {
        this.utils = utils;
        this.account = account;
        this.tokenAccounts = tokenAccounts;
    }

    static makeOrders(orders: [number, number][]): Order[] {
        return orders.map(([price, size]) => ({ price, size }));
    }

    async placeOrders(market: Market, bids: Order[], asks: Order[]) {
        const baseTokenAccount =
            this.tokenAccounts[market.baseMintAddress.toBase58()];

        const quoteTokenAccount =
            this.tokenAccounts[market.quoteMintAddress.toBase58()];

        const askOrderTxs = [];
        const bidOrderTxs = [];

        const placeOrderDefaultParams = {
            owner: this.account.publicKey,
            clientId: undefined,
            openOrdersAddressKey: undefined,
            openOrdersAccount: undefined,
            feeDiscountPubkey: null,
        };

        for (const entry of asks) {
            const { transaction, signers } =
                await market.makePlaceOrderTransaction(
                    this.utils.connection(),
                    {
                        payer: baseTokenAccount,
                        side: "sell",
                        price: entry.price,
                        size: entry.size,
                        orderType: "postOnly",
                        selfTradeBehavior: "abortTransaction",
                        ...placeOrderDefaultParams,
                    }
                );

            askOrderTxs.push([transaction, [this.account, ...signers]]);
        }

        for (const entry of bids) {
            const { transaction, signers } =
                await market.makePlaceOrderTransaction(
                    this.utils.connection(),
                    {
                        payer: quoteTokenAccount,
                        side: "buy",
                        price: entry.price,
                        size: entry.size,
                        orderType: "postOnly",
                        selfTradeBehavior: "abortTransaction",
                        ...placeOrderDefaultParams,
                    }
                );

            bidOrderTxs.push([transaction, [this.account, ...signers]]);
        }

        await this.utils.sendAndConfirmTransactionSet(
            ...askOrderTxs,
            ...bidOrderTxs
        );
    }
}