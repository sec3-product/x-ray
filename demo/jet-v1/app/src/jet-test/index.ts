/**
 * Utilities for writing integration tests
 *
 * @module
 */

import { Wallet, BN } from "@project-serum/anchor";
import { NATIVE_MINT, Token, TOKEN_PROGRAM_ID } from "@solana/spl-token";
import {
    Connection,
    Keypair,
    PublicKey,
    sendAndConfirmTransaction,
    SystemProgram,
    Transaction,
} from "@solana/web3.js";
import { PythUtils } from "./pyth";

export class TestUtils {
    public static readonly pythProgramId = PythUtils.programId;

    public pyth: PythUtils;

    private conn: Connection;
    private wallet: Wallet;
    private authority: Keypair;

    constructor(conn: Connection, funded: Wallet) {
        this.conn = conn;
        this.wallet = funded;
        this.authority = this.wallet.payer;
        this.pyth = new PythUtils(conn, funded);
    }

    /**
     * Create a new SPL token
     * @param decimals The number of decimals for the token.
     * @param authority The account with authority to mint/freeze tokens.
     * @returns The new token
     */
    async createToken(
        decimals: number,
        authority: PublicKey = this.authority.publicKey
    ): Promise<TestToken> {
        const token = await Token.createMint(
            this.conn,
            this.authority,
            authority,
            authority,
            decimals,
            TOKEN_PROGRAM_ID
        );

        return new TestToken(this.conn, token, decimals);
    }

    async createNativeToken() {
        const token = new Token(this.conn, NATIVE_MINT, TOKEN_PROGRAM_ID, this.authority);

        return new TestToken(this.conn, token, 9);
    }

    /**
     * Create a new wallet with some initial funding.
     * @param lamports The amount of lamports to fund the wallet account with.
     * @returns The keypair for the new wallet.
     */
    async createWallet(lamports: number): Promise<Keypair> {
        const wallet = Keypair.generate();
        const fundTx = new Transaction().add(
            SystemProgram.transfer({
                fromPubkey: this.wallet.publicKey,
                toPubkey: wallet.publicKey,
                lamports,
            })
        );

        await sendAndConfirmTransaction(this.conn, fundTx, [this.authority]);
        return wallet;
    }

    /**
     * Create a new token account with some initial funding.
     * @param token The token to create an account for
     * @param owner The account that should own these tokens
     * @param amount The initial amount of tokens to provide as funding
     * @returns The address for the created account
     */
    async createTokenAccount(
        token: Token,
        owner: PublicKey,
        amount: BN
    ): Promise<PublicKey> {
        if(token.publicKey == NATIVE_MINT) {
            const account = await Token.createWrappedNativeAccount(this.conn, TOKEN_PROGRAM_ID, owner, this.authority, amount.toNumber());
            return account;
        } else {
            const account = await token.createAccount(owner);
            await token.mintTo(account, this.authority, [], amount.toNumber());
            return account;
        }
    }

    /**
     * Find a program derived address
     * @param programId The program the address is being derived for
     * @param seeds The seeds to find the address
     * @returns The address found and the bump seed required
     */
    async findProgramAddress(
        programId: PublicKey,
        seeds: (HasPublicKey | ToBytes | Uint8Array | string)[]
    ): Promise<[PublicKey, number]> {
        const seed_bytes = seeds.map((s) => {
            if (typeof s == "string") {
                return Buffer.from(s);
            } else if ("publicKey" in s) {
                return s.publicKey.toBytes();
            } else if ("toBytes" in s) {
                return s.toBytes();
            } else {
                return s;
            }
        });
        return await PublicKey.findProgramAddress(seed_bytes, programId);
    }
}

/**
 * Convert some value/object to use `BN` type to represent numbers.
 *
 * If the value is a number, its converted to a `BN`. If the value is
 * an object, then each field is (recursively) converted to a `BN`.
 *
 * @param obj The value or object to convert.
 * @returns The object as a`BN`
 */
export function toBN(obj: any): any {
    if (typeof obj == "number") {
        return new BN(obj);
    } else if (typeof obj == "object") {
        const bnObj: any = {};

        for (const field in obj) {
            bnObj[field] = toBN(obj[field]);
        }

        return bnObj;
    }

    return obj;
}

/**
 * Convert some object of fields with address-like values,
 * such that the values are converted to their `PublicKey` form.
 * @param obj The object to convert
 */
export function toPublicKeys(obj: Record<string, string | PublicKey | HasPublicKey>): any {
    const newObj: Record<string, string | PublicKey | HasPublicKey> = {};

    for (const key in obj) {
        const value = obj[key];

        if (typeof value == "string") {
            newObj[key] = new PublicKey(value);
        } else if ('publicKey' in value) {
            newObj[key] = value.publicKey;
        } else {
            newObj[key] = value;
        }
    }

    return newObj;
}

/**
 * Convert some object of fields with address-like values,
 * such that the values are converted to their base58 `PublicKey` form.
 * @param obj The object to convert
 */
export function toBase58(obj: Record<string, string | PublicKey | HasPublicKey>): any {
    const newObj: Record<string, string | PublicKey | HasPublicKey> = {};

    for (const key in obj) {
        const value = obj[key];

        if(value == undefined) {
            continue;
        } else if (typeof value == "string") {
            newObj[key] = value;
        } else if ('publicKey' in value) {
            newObj[key] = value.publicKey.toBase58();
        } else if ('toBase58' in value && typeof value.toBase58 == "function") {
            newObj[key] = value.toBase58();
        } else {
            newObj[key] = value;
        }
    }

    return newObj;
}

export class TestToken extends Token {
    decimals: number;

    constructor(conn: Connection, token: Token, decimals: number) {
        super(conn, token.publicKey, token.programId, token.payer);
        this.decimals = decimals;
    }


    /**
     * Convert a token amount to the integer format for the mint
     * @param token The token mint
     * @param amount The amount of tokens
     */
    amount(amount: BN | number): BN {
        if (typeof amount == "number") {
            amount = new BN(amount);
        }

        const one_unit = new BN(10).pow(new BN(this.decimals));
        return amount.mul(one_unit);
    }
}

interface ToBytes {
    toBytes(): Uint8Array;
}

interface HasPublicKey {
    publicKey: PublicKey;
}